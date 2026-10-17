"""Smart Resume Ranker — premium Streamlit dashboard.

Workflow:
  1. Upload JD (+ optional paste) and multiple resumes (PDF/DOCX).
  2. In-memory extraction (no disk writes) -> spaCy preprocessing.
  3. Sparse TF-IDF + cosine similarity ranking.
  4. Interactive rankings, distribution chart, skill gaps, CSV export.

Theming
-------
Streamlit 1.66 does **not** expose a ``data-theme`` attribute in the DOM and has
retired the old BaseWeb test-ids, so the previous attribute-based CSS was dead.
Instead this app resolves the active theme in Python and injects the whole
colour palette as CSS custom properties (``:root { … }``) on every run, while
Streamlit's native ``theme.base`` is switched in lock-step. Everything then
recolours correctly from a single source of truth.
"""

from __future__ import annotations

import io
import logging
import time
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

from utils.matcher import RankedCandidate, rank_resumes
from utils.parser import (
    ALLOWED_EXTENSIONS,
    MAX_FILE_SIZE_BYTES,
    EncryptedDocumentError,
    FileValidationError,
    ScannedDocumentError,
    TextExtractionError,
    extract_text,
    sanitize_filename,
)
from utils.preprocessor import get_nlp, preprocess_corpus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Smart Resume Ranker",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

MAX_RESUMES = 50

# ═══════════════════════════════════════════════════════════════════════════════
#  THEME PALETTES
#  Each palette is a flat dict of CSS custom properties. The resolved palette is
#  serialised into a `:root { … }` block and injected on every run, so switching
#  the theme recolours the *entire* UI with zero reliance on DOM attributes.
# ═══════════════════════════════════════════════════════════════════════════════

PALETTES: Dict[str, Dict[str, str]] = {
    "light": {
        # surfaces
        "--bg":          "#F6F6FB",
        "--bg-soft":     "#EFEFF6",
        "--surface":     "#FFFFFF",
        "--surface-2":   "#F6F6FB",
        "--border":      "rgba(17,17,34,.09)",
        "--border-2":    "rgba(17,17,34,.16)",
        # text
        "--tx":          "#0B0B12",
        "--tx-2":        "#37374A",
        "--tx-3":        "#6A6A7B",
        "--tx-4":        "#9C9CAC",
        # brand
        "--brand":       "#7C3AED",
        "--brand-2":     "#0EA5E9",
        "--brand-ink":   "#6D28D9",
        "--brand-glow":  "rgba(124,58,237,.28)",
        "--grad-brand":  "linear-gradient(135deg,#7C3AED,#2563EB)",
        "--grad-kpi":    "linear-gradient(135deg,#7C3AED,#2563EB)",
        # elevation
        "--shadow-sm":   "0 1px 2px rgba(18,12,42,.06)",
        "--shadow":      "0 6px 22px rgba(18,12,42,.07), 0 1px 3px rgba(18,12,42,.05)",
        "--shadow-lg":   "0 18px 48px rgba(18,12,42,.11), 0 6px 16px rgba(18,12,42,.06)",
        # dropzone
        "--drop-bg":     "#F5F3FF",
        "--drop-bd":     "#DDD6FE",
        "--drop-bd-hv":  "#A78BFA",
        "--drop-bg-hv":  "#EDE9FE",
        # sidebar
        "--sidebar-bg":  "linear-gradient(180deg,#FFFFFF 0%,#F4F4FB 100%)",
        "--sidebar-bd":  "rgba(17,17,34,.08)",
        "--sb-card":     "#FFFFFF",
        "--sb-card-bd":  "rgba(17,17,34,.08)",
        # tags
        "--tag-bg":      "rgba(124,58,237,.09)",
        "--tag-fg":      "#5B21B6",
        "--tag-bd":      "rgba(124,58,237,.22)",
        # badges
        "--b-green-bg":  "#DCFCE7", "--b-green-fg":  "#15803D", "--b-green-bd":  "#BBF7D0",
        "--b-violet-bg": "#EDE9FE", "--b-violet-fg": "#5B21B6", "--b-violet-bd": "#C4B5FD",
        "--b-amber-bg":  "#FEF3C7", "--b-amber-fg":  "#92400E", "--b-amber-bd":  "#FDE68A",
        "--b-rose-bg":   "#FFF1F2", "--b-rose-fg":   "#BE123C", "--b-rose-bd":   "#FECDD3",
    },
    "dark": {
        "--bg":          "#0A0A11",
        "--bg-soft":     "#101019",
        "--surface":     "rgba(255,255,255,.045)",
        "--surface-2":   "rgba(255,255,255,.07)",
        "--border":      "rgba(255,255,255,.09)",
        "--border-2":    "rgba(255,255,255,.17)",
        "--tx":          "#F4F4F8",
        "--tx-2":        "#D2D2DC",
        "--tx-3":        "#A0A0AE",
        "--tx-4":        "#6E6E7C",
        "--brand":       "#A78BFA",
        "--brand-2":     "#38BDF8",
        "--brand-ink":   "#C4B5FD",
        "--brand-glow":  "rgba(124,58,237,.50)",
        "--grad-brand":  "linear-gradient(135deg,#7C3AED,#2563EB)",
        "--grad-kpi":    "linear-gradient(135deg,#A78BFA,#38BDF8)",
        "--shadow-sm":   "0 1px 2px rgba(0,0,0,.45)",
        "--shadow":      "0 8px 26px rgba(0,0,0,.45)",
        "--shadow-lg":   "0 22px 60px rgba(0,0,0,.55)",
        "--drop-bg":     "rgba(124,58,237,.13)",
        "--drop-bd":     "rgba(124,58,237,.42)",
        "--drop-bd-hv":  "#7C3AED",
        "--drop-bg-hv":  "rgba(124,58,237,.22)",
        "--sidebar-bg":  "linear-gradient(180deg,#0D0A1A 0%,#07060F 100%)",
        "--sidebar-bd":  "rgba(255,255,255,.07)",
        "--sb-card":     "rgba(255,255,255,.035)",
        "--sb-card-bd":  "rgba(255,255,255,.09)",
        "--tag-bg":      "rgba(124,58,237,.16)",
        "--tag-fg":      "#C4B5FD",
        "--tag-bd":      "rgba(124,58,237,.36)",
        "--b-green-bg":  "#052E16", "--b-green-fg":  "#86EFAC", "--b-green-bd":  "#166534",
        "--b-violet-bg": "#2E1065", "--b-violet-fg": "#C4B5FD", "--b-violet-bd": "#4C1D95",
        "--b-amber-bg":  "#431407", "--b-amber-fg":  "#FCD34D", "--b-amber-bd":  "#92400E",
        "--b-rose-bg":   "#4C0519", "--b-rose-fg":   "#FDA4AF", "--b-rose-bd":   "#9F1239",
    },
}

# Native Streamlit theme colours, applied through the (private) config API so
# every built-in widget / chart / table follows the active mode.
NATIVE_THEME: Dict[str, Dict[str, str]] = {
    "light": {
        "theme.base":                    "light",
        "theme.primaryColor":            "#7C3AED",
        "theme.backgroundColor":         "#F6F6FB",
        "theme.secondaryBackgroundColor": "#FFFFFF",
        "theme.textColor":               "#0B0B12",
    },
    "dark": {
        "theme.base":                    "dark",
        "theme.primaryColor":            "#A78BFA",
        "theme.backgroundColor":         "#0A0A11",
        "theme.secondaryBackgroundColor": "#14141F",
        "theme.textColor":               "#F4F4F8",
    },
}

_THEME_LABELS = {"Light": "☀️  Light", "Auto": "💻  Auto", "Dark": "🌙  Dark"}


# ═══════════════════════════════════════════════════════════════════════════════
#  STYLESHEET TEMPLATE  ( `/*__TOKENS__*/` is replaced with the active palette )
# ═══════════════════════════════════════════════════════════════════════════════
_CSS_TEMPLATE = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── TOKENS (injected per-run) ─────────────────────────────────────────── */
:root {
/*__TOKENS__*/
  --font:      'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
  --font-mono: 'JetBrains Mono', 'Fira Code', monospace;
  --radius-sm: 10px;
  --radius:    14px;
  --radius-lg: 20px;
  --radius-xl: 26px;
  --grad-hero: linear-gradient(135deg,#1E0A3C 0%,#3B0764 42%,#0C4A6E 100%);
  --grad-spot: linear-gradient(135deg,#1E0A3C 0%,#4C1D95 52%,#0C4A6E 100%);
}

/* ── BASE ──────────────────────────────────────────────────────────────── */
html, body, [class*="css"] {
  font-family: var(--font);
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
  background: var(--bg) !important;
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stMainBlockContainer"] {
  max-width: 1320px;
  padding: 0 2rem 5rem;
}
[data-testid="stVerticalBlock"] { gap: .9rem; }
a { color: var(--brand-ink); }

/* ── CHROME CLEANUP ────────────────────────────────────────────────────── */
[data-testid="stMainMenu"], [data-testid="stMainMenuButton"],
[data-testid="stToolbarActions"], [data-testid="stAppDeployButton"],
div[class*="stAppDeployButton"] { display: none !important; }
#MainMenu { visibility: hidden !important; }
footer { visibility: hidden !important; }
[data-testid="stDecoration"], [data-testid="stStatusWidget"] { display: none !important; }

/* Sidebar expand button */
[data-testid="stSidebarCollapseButton"],
[data-testid="stExpandSidebarButton"] {
  visibility: visible !important; opacity: 1 !important; pointer-events: auto !important;
}
[data-testid="stExpandSidebarButton"] {
  background: var(--grad-brand) !important; color: #fff !important;
  border-radius: 10px !important; margin: .5rem !important; padding: .22rem .4rem !important;
  box-shadow: 0 4px 14px var(--brand-glow) !important;
}
[data-testid="stExpandSidebarButton"] svg { color: #fff !important; }

/* ── SIDEBAR ───────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: var(--sidebar-bg) !important;
  border-right: 1px solid var(--sidebar-bd) !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] .stMarkdown { color: var(--tx-2); }
[data-testid="stSidebar"] label { color: var(--tx-3) !important; font-size: .81rem !important; font-weight: 600 !important; }
[data-testid="stSidebar"] hr { border-color: var(--sidebar-bd) !important; margin: .7rem 0 !important; }

/* side cards (containers carrying a `side_` key) */
[data-testid="stVerticalBlock"][class*="st-key-side_"] {
  background: var(--sb-card) !important;
  border: 1px solid var(--sb-card-bd) !important;
  border-radius: var(--radius-lg) !important;
  box-shadow: none !important;
  padding: 1rem 1.05rem !important;
}

/* Sidebar brand */
.sb-logo { display: flex; align-items: center; gap: .75rem; padding: .15rem 0 .55rem; }
.sb-logo-mark {
  width: 42px; height: 42px; border-radius: 13px; flex-shrink: 0;
  background: var(--grad-brand);
  display: flex; align-items: center; justify-content: center; font-size: 1.2rem;
  box-shadow: 0 8px 20px var(--brand-glow);
}
.sb-logo-text-main { font-size: .98rem; font-weight: 800; color: var(--tx) !important; letter-spacing: -.02em; line-height: 1.15; }
.sb-logo-text-sub  { font-size: .68rem; color: var(--tx-4) !important; letter-spacing: .09em; text-transform: uppercase; font-weight: 600; }

.sb-cap {
  font-size: .62rem; font-weight: 700; letter-spacing: .15em; text-transform: uppercase;
  color: var(--brand-ink) !important;
  margin: .15rem 0 .55rem; display: flex; align-items: center; gap: .45rem;
}
.sb-cap::before { content: ''; width: 15px; height: 2px; border-radius: 2px; background: var(--grad-brand); }

.sb-info { font-size: .77rem; line-height: 2.05; color: var(--tx-3); }
.sb-info b { color: var(--tx-2); font-weight: 700; }

/* Developer card */
.dev-card {
  padding: 1rem 1.1rem; border-radius: 18px;
  background: rgba(124,58,237,.12);
  border: 1px solid rgba(124,58,237,.26);
  background: linear-gradient(135deg, color-mix(in srgb, var(--brand) 15%, transparent), color-mix(in srgb, var(--brand-2) 10%, transparent));
  border: 1px solid color-mix(in srgb, var(--brand) 28%, transparent);
}
.dev-av {
  width: 42px; height: 42px; border-radius: 50%; background: var(--grad-brand);
  display: flex; align-items: center; justify-content: center; font-size: 1.2rem; margin-bottom: .55rem;
  box-shadow: 0 6px 16px var(--brand-glow);
}
.dev-name { font-size: .9rem; font-weight: 800; color: var(--tx) !important; letter-spacing: -.01em; }
.dev-role { font-size: .71rem; color: var(--tx-3) !important; margin: .06rem 0 .65rem; }
.dev-btn {
  display: inline-flex; align-items: center; gap: .35rem;
  background: var(--grad-brand); color: #fff !important; text-decoration: none !important;
  font-size: .75rem; font-weight: 700; padding: .4rem .95rem; border-radius: 10px;
  box-shadow: 0 4px 14px var(--brand-glow); transition: filter .15s, box-shadow .15s;
}
.dev-btn:hover { filter: brightness(1.09); box-shadow: 0 7px 22px var(--brand-glow); }
.dev-btn svg { width: 11px; height: 11px; }

/* ── HERO ──────────────────────────────────────────────────────────────── */
.hero {
  position: relative; overflow: hidden;
  background: var(--grad-hero); border: 1px solid rgba(255,255,255,.10);
  border-radius: var(--radius-xl); padding: 2.6rem 3rem;
  margin: .3rem 0 2rem;
  box-shadow: 0 30px 74px rgba(28,3,80,.50);
  display: block;
}
@media (max-width: 980px) { .hero { padding: 2.2rem 1.6rem; } }
.hero::before {
  content: ''; position: absolute; top: -130px; right: -70px; width: 500px; height: 500px;
  border-radius: 50%; background: radial-gradient(circle, rgba(14,165,233,.24) 0%, transparent 68%);
  pointer-events: none;
}
.hero::after {
  content: ''; position: absolute; bottom: -150px; left: 26%; width: 430px; height: 430px;
  border-radius: 50%; background: radial-gradient(circle, rgba(124,58,237,.20) 0%, transparent 68%);
  pointer-events: none;
}
.hero-inner { position: relative; z-index: 1; }
.hero-badge {
  display: inline-flex; align-items: center; gap: .45rem;
  background: rgba(255,255,255,.10); border: 1px solid rgba(255,255,255,.18);
  color: rgba(255,255,255,.9); font-size: .7rem; font-weight: 700;
  letter-spacing: .1em; text-transform: uppercase;
  padding: .3rem .9rem; border-radius: 999px; margin-bottom: 1.05rem;
}
.hero-badge .pulse {
  width: 7px; height: 7px; border-radius: 50%; background: #38BDF8;
  box-shadow: 0 0 8px #38BDF8; animation: hpulse 2.2s ease-in-out infinite;
}
@keyframes hpulse { 0%,100%{ opacity:1; transform:scale(1); } 50%{ opacity:.45; transform:scale(.72); } }
.hero h1 { color: #fff !important; font-size: 2.45rem; font-weight: 800; margin: 0 0 .75rem; letter-spacing: -.033em; line-height: 1.13; }
.hero h1 em { font-style: normal; color: #7DD3FC; }
.hero p { color: rgba(255,255,255,.74); font-size: .97rem; line-height: 1.72; margin: 0 0 1.5rem; max-width: 640px; }
.hero-chips { display: flex; gap: .5rem; flex-wrap: wrap; }
.hero-chip {
  display: inline-flex; align-items: center; gap: .3rem;
  background: rgba(255,255,255,.09); border: 1px solid rgba(255,255,255,.17);
  color: rgba(255,255,255,.86); font-size: .73rem; font-weight: 600;
  padding: .3rem .85rem; border-radius: 999px;
}

/* ── SECTION LABELS ────────────────────────────────────────────────────── */
.sec-cap {
  display: inline-flex; align-items: center; gap: .5rem;
  font-size: .65rem; font-weight: 700; letter-spacing: .16em; text-transform: uppercase;
  color: var(--brand-ink); margin: 1.6rem 0 .35rem;
}
.sec-cap::before { content: ''; width: 20px; height: 2px; border-radius: 2px; background: linear-gradient(90deg, var(--brand), var(--brand-2)); }
.sec-title { font-size: 1.45rem; font-weight: 800; color: var(--tx); margin: 0 0 .3rem; letter-spacing: -.027em; }
.sec-sub { color: var(--tx-3); font-size: .89rem; line-height: 1.65; margin-bottom: 1.25rem; }

/* ── MAIN CARDS ────────────────────────────────────────────────────────── */
[data-testid="stVerticalBlock"][class*="st-key-card_"],
[data-testid="stVerticalBlock"][class*="st-key-kpi_"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-lg) !important;
  box-shadow: var(--shadow) !important;
  padding: 1.2rem 1.25rem !important;
  transition: box-shadow .25s, border-color .25s, transform .25s !important;
}
[data-testid="stVerticalBlock"][class*="st-key-card_"]:hover,
[data-testid="stVerticalBlock"][class*="st-key-kpi_"]:hover {
  box-shadow: var(--shadow-lg) !important;
  border-color: var(--brand-glow) !important;
  border-color: color-mix(in srgb, var(--brand) 32%, transparent) !important;
  transform: translateY(-2px);
}

.uc-head { display: flex; align-items: center; gap: .6rem; font-size: .95rem; font-weight: 700; color: var(--tx); margin-bottom: .3rem; }
.uc-icon { width: 32px; height: 32px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: .92rem; flex-shrink: 0; }
.uc-violet { background: linear-gradient(135deg,#7C3AED,#A78BFA); box-shadow: 0 4px 12px rgba(124,58,237,.35); }
.uc-sky    { background: linear-gradient(135deg,#0369A1,#0EA5E9); box-shadow: 0 4px 12px rgba(14,165,233,.35); }
.uc-hint   { font-size: .72rem; color: var(--tx-3); font-weight: 600; }

/* ── FILE DROP ZONES ───────────────────────────────────────────────────── */
[data-testid="stFileUploaderDropzone"] {
  min-height: 130px;
  border: 1.5px dashed var(--drop-bd) !important;
  border-radius: var(--radius) !important;
  background: var(--drop-bg) !important;
  transition: border-color .2s, background .2s;
}
[data-testid="stFileUploaderDropzone"]:hover {
  border-color: var(--drop-bd-hv) !important;
  background: var(--drop-bg-hv) !important;
}
[data-testid="stFileUploaderDropzone"] svg { color: var(--brand) !important; }

/* ── KPI ───────────────────────────────────────────────────────────────── */
.kpi { text-align: center; padding: .35rem .3rem; }
.kpi-ico { font-size: 1.45rem; margin-bottom: .35rem; line-height: 1; }
.kpi-lbl { font-size: .63rem; font-weight: 700; letter-spacing: .13em; text-transform: uppercase; color: var(--tx-3); }
.kpi-val {
  font-size: 2rem; font-weight: 800; line-height: 1.1; margin: .22rem 0 .14rem;
  background: var(--grad-kpi); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.kpi-sub { font-size: .74rem; color: var(--tx-3); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

/* ── SPOTLIGHT ─────────────────────────────────────────────────────────── */
.spotlight {
  position: relative; overflow: hidden; background: var(--grad-spot);
  border: 1px solid rgba(255,255,255,.10); border-radius: 22px; padding: 1.6rem 2rem; color: #fff;
  box-shadow: 0 20px 52px rgba(28,3,80,.42); margin: 1.35rem 0 1.15rem;
  display: flex; align-items: center; justify-content: space-between; gap: 1.2rem; flex-wrap: wrap;
}
.spotlight::before {
  content: ''; position: absolute; top: -70px; right: -50px; width: 320px; height: 320px; border-radius: 50%;
  background: radial-gradient(circle, rgba(14,165,233,.24) 0%, transparent 65%); pointer-events: none;
}
.sp-left { position: relative; z-index: 1; }
.sp-right { position: relative; z-index: 1; display: flex; flex-direction: column; align-items: flex-end; gap: .5rem; }
.sp-eyebrow { display: flex; align-items: center; gap: .4rem; font-size: .63rem; font-weight: 700; letter-spacing: .13em; text-transform: uppercase; color: rgba(255,255,255,.58); margin-bottom: .35rem; }
.sp-eyebrow::before { content: ''; width: 7px; height: 7px; border-radius: 50%; background: #38BDF8; box-shadow: 0 0 10px #38BDF8; }
.sp-name { font-size: 1.45rem; font-weight: 800; margin: 0 0 .3rem; letter-spacing: -.027em; }
.sp-sub { font-size: .85rem; color: rgba(255,255,255,.74); line-height: 1.55; }
.sp-score {
  font-size: 2rem; font-weight: 800; background: rgba(255,255,255,.14);
  border: 1px solid rgba(255,255,255,.22); border-radius: 16px; padding: .45rem 1.1rem; line-height: 1; white-space: nowrap;
}

/* ── BADGES & TAGS ─────────────────────────────────────────────────────── */
.badge { display: inline-block; font-weight: 700; font-size: .76rem; padding: .22rem .75rem; border-radius: 999px; letter-spacing: .01em; }
.b-green  { background: var(--b-green-bg);  color: var(--b-green-fg);  border: 1px solid var(--b-green-bd); }
.b-violet { background: var(--b-violet-bg); color: var(--b-violet-fg); border: 1px solid var(--b-violet-bd); }
.b-amber  { background: var(--b-amber-bg);  color: var(--b-amber-fg);  border: 1px solid var(--b-amber-bd); }
.b-rose   { background: var(--b-rose-bg);   color: var(--b-rose-fg);   border: 1px solid var(--b-rose-bd); }

.tag {
  display: inline-flex; align-items: center; background: var(--tag-bg); color: var(--tag-fg);
  border: 1px solid var(--tag-bd); font-size: .73rem; font-weight: 600; font-family: var(--font-mono);
  padding: .18rem .65rem; border-radius: 8px; margin: .1rem; transition: filter .15s;
}
.tag:hover { filter: brightness(1.08); }

/* ── STEPS ─────────────────────────────────────────────────────────────── */
.step { padding: .35rem .15rem; }
.step-n { font-size: .62rem; font-weight: 700; letter-spacing: .13em; text-transform: uppercase; color: var(--brand-ink); margin-bottom: .5rem; display: flex; align-items: center; gap: .45rem; }
.step-n::after { content: ''; flex: 1; height: 1px; background: var(--border); }
.step-icon { font-size: 1.8rem; display: block; margin-bottom: .5rem; }
.step-t { font-size: .98rem; font-weight: 700; color: var(--tx); margin-bottom: .35rem; }
.step-b { font-size: .83rem; color: var(--tx-3); line-height: 1.68; }

/* ── BUTTONS ───────────────────────────────────────────────────────────── */
div.stButton > button[kind="primary"] {
  background: var(--grad-brand) !important; border: none !important;
  border-radius: var(--radius) !important; padding: .85rem 1.4rem !important;
  font-family: var(--font) !important; font-weight: 700 !important; font-size: 1.02rem !important;
  letter-spacing: .01em !important; color: #fff !important;
  box-shadow: 0 10px 30px var(--brand-glow) !important;
  transition: filter .2s, box-shadow .2s, transform .12s !important;
}
div.stButton > button[kind="primary"]:hover {
  filter: brightness(1.08) !important; box-shadow: 0 16px 40px var(--brand-glow) !important; transform: translateY(-1px) !important;
}
div.stButton > button[kind="primary"]:active { transform: translateY(0) !important; }
div.stButton > button[kind="secondary"] {
  background: var(--surface) !important; border: 1px solid var(--border-2) !important;
  border-radius: var(--radius) !important; color: var(--tx) !important; font-weight: 600 !important;
}
div.stDownloadButton > button {
  border-radius: var(--radius) !important; font-weight: 700 !important; font-family: var(--font) !important;
  border: 1.5px solid var(--brand) !important; color: var(--brand-ink) !important;
  background: transparent !important; transition: background .15s, color .15s !important;
}
div.stDownloadButton > button:hover { background: var(--brand) !important; color: #fff !important; }

/* ── TABS (React-Aria, Streamlit 1.66) ─────────────────────────────────── */
[data-testid="stTabs"] [role="tablist"] { gap: .3rem; border-bottom: 1.5px solid var(--border) !important; }
[data-testid="stTab"] {
  border-radius: 10px 10px 0 0 !important; padding: .55rem 1.1rem !important;
  font-family: var(--font) !important; font-weight: 600 !important; font-size: .87rem !important;
  color: var(--tx-3) !important; transition: color .15s, background .15s;
}
[data-testid="stTab"]:hover { color: var(--tx) !important; background: var(--surface-2) !important; }
[data-testid="stTab"][aria-selected="true"] { color: var(--brand-ink) !important; }
[data-testid="stTab"] p { color: inherit !important; }

/* ── EXPANDERS ─────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
  border: 1px solid var(--border) !important; border-radius: var(--radius) !important;
  background: var(--surface) !important; overflow: hidden; transition: border-color .2s !important;
}
[data-testid="stExpander"]:hover { border-color: color-mix(in srgb, var(--brand) 30%, transparent) !important; }
[data-testid="stExpander"] summary { font-weight: 600 !important; }
[data-testid="stExpander"] summary:hover { color: var(--brand-ink) !important; }

/* ── PROGRESS ──────────────────────────────────────────────────────────── */
[data-testid="stProgressBar"] > div { background: var(--surface-2) !important; border-radius: 999px !important; overflow: hidden; }
[data-testid="stProgressBar"] > div > div { background: var(--grad-brand) !important; border-radius: 999px !important; }

/* ── MISC WIDGET POLISH ────────────────────────────────────────────────── */
[data-testid="stTextArea"] textarea {
  border-radius: var(--radius-sm) !important; font-family: var(--font) !important; font-size: .87rem !important;
  background: var(--surface) !important; color: var(--tx) !important;
}
[data-testid="stAlert"] { border-radius: var(--radius) !important; }
[data-testid="stCaptionContainer"] p { color: var(--tx-3) !important; font-size: .82rem; }
hr { border-color: var(--border) !important; }

/* Appearance selector (segmented control, `theme_seg` key) */
[data-testid="stElementContainer"][class*="st-key-theme_seg"] { margin-bottom: .15rem; width: 100% !important; }
[class*="st-key-theme_seg"] .stButtonGroup,
[class*="st-key-theme_seg"] [role="radiogroup"] { width: 100% !important; max-width: 100% !important; }
[class*="st-key-theme_seg"] [role="radiogroup"] { gap: 0 !important; }
[class*="st-key-theme_seg"] button {
  flex: 1 1 0 !important;
  display: flex !important; align-items: center; justify-content: center;
  font-family: var(--font) !important; font-weight: 600 !important; font-size: .85rem !important;
  border-radius: 9px !important; padding: 0 .4rem !important; min-height: 36px;
}
[class*="st-key-theme_seg"] button * { font-size: inherit !important; }
[class*="st-key-theme_seg"] button p { margin: 0 !important; line-height: 1.1 !important; }
[class*="st-key-theme_seg"] button [data-testid="stIconEmoji"] { font-size: 1.08em !important; }

/* ── FOOTER ────────────────────────────────────────────────────────────── */
.footer {
  text-align: center; color: var(--tx-3); font-size: .75rem;
  padding: 2rem 0 .5rem; letter-spacing: .01em; border-top: 1px solid var(--border); margin-top: 2rem;
}
.footer a { color: var(--brand-ink); text-decoration: none; font-weight: 600; }
.footer a:hover { text-decoration: underline; }
"""


def _css_for(base: str) -> str:
    """Return the stylesheet with the resolved palette baked into ``:root``."""
    tokens = "\n".join(f"  {k}: {v};" for k, v in PALETTES[base].items())
    return _CSS_TEMPLATE.replace("/*__TOKENS__*/", tokens)


# ═══════════════════════════════════════════════════════════════════════════════
#  THEME ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
if "theme_mode" not in st.session_state:
    st.session_state["theme_mode"] = "Auto"

# Capture the OS colour-scheme once per session — ``st.context.theme.type`` is
# authoritative on the first run, and this caches it so Auto keeps following the
# OS even after the user toggles Light/Dark manually.
if "_os_scheme" not in st.session_state:
    try:
        _t = st.context.theme.type
    except Exception:  # pragma: no cover - very old Streamlit
        _t = None
    st.session_state["_os_scheme"] = _t if _t in ("light", "dark") else "light"


def resolve_base() -> str:
    """Map the Light/Auto/Dark selector to a concrete 'light' | 'dark' base."""
    mode = st.session_state.get("theme_mode", "Auto")
    if mode == "Light":
        return "light"
    if mode == "Dark":
        return "dark"
    return st.session_state.get("_os_scheme", "light")


def apply_native_theme(base: str) -> None:
    """Switch Streamlit's native theme, rerunning once when it actually changes."""
    if st.session_state.get("_applied_base") == base:
        return
    try:
        changed = False
        for key, value in NATIVE_THEME[base].items():
            try:
                if st.get_option(key) != value:
                    changed = True
                st._config.set_option(key, value)  # noqa: SLF001
            except Exception:
                logger.warning("Could not set theme option %s", key)
        st.session_state["_applied_base"] = base
        if changed:
            st.rerun()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Native theme switch unavailable: %s", exc)
        st.session_state["_theme_fallback"] = True


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def _badge(score: float) -> str:
    if score >= 75:
        return f'<span class="badge b-green">{score:.1f}%</span>'
    if score >= 50:
        return f'<span class="badge b-violet">{score:.1f}%</span>'
    if score >= 25:
        return f'<span class="badge b-amber">{score:.1f}%</span>'
    return f'<span class="badge b-rose">{score:.1f}%</span>'


@st.cache_resource(show_spinner="Loading language model…")
def _load_nlp_cached() -> object:
    return get_nlp()


def _read_upload(uf: object) -> Tuple[str, bytes]:
    name = sanitize_filename(getattr(uf, "name", "unnamed"))
    try:
        data: bytes = uf.getvalue()  # type: ignore[attr-defined]
    except Exception as exc:
        raise ValueError(f"Could not read '{name}': {exc}") from exc
    if len(data) > MAX_FILE_SIZE_BYTES:
        raise ValueError(
            f"'{name}' is {len(data)/1_048_576:.1f} MB "
            f"(limit {MAX_FILE_SIZE_BYTES//1_048_576} MB)."
        )
    if not data:
        raise ValueError(f"'{name}' is empty.")
    return name, data


def _process_batch(
    jd_text: str, resume_files: list, remove_pii: bool,
    max_features: int, top_gaps: int,
) -> Tuple[List[RankedCandidate], dict, List[Tuple[str, str]], List[Tuple[str, str]]]:
    t0 = time.perf_counter()
    names: List[str] = []
    raw_texts: List[str] = []
    failures: List[Tuple[str, str]] = []
    scanned: List[Tuple[str, str]] = []

    for uf in resume_files:
        orig = getattr(uf, "name", "unknown")
        try:
            safe_name, data = _read_upload(uf)
            try:
                text = extract_text(data, safe_name)
            except ScannedDocumentError as exc:
                scanned.append((orig, str(exc))); continue
            except EncryptedDocumentError as exc:
                failures.append((orig, f"Encrypted: {exc}")); continue
            except (FileValidationError, TextExtractionError) as exc:
                failures.append((orig, str(exc))); continue
            finally:
                del data
            if len(text.strip()) < 10:
                scanned.append((orig, "Almost no selectable text — likely scanned.")); continue
            names.append(safe_name)
            raw_texts.append(text)
        except ValueError as exc:
            failures.append((orig, str(exc)))
        except Exception as exc:
            logger.exception("Unexpected error for %s", orig)
            failures.append((orig, f"Unexpected: {exc}"))

    if not names:
        raise ValueError(
            "No valid resumes could be extracted. "
            "Files may be scanned, encrypted or corrupted."
        )

    try:
        processed = preprocess_corpus([jd_text] + raw_texts, remove_pii=remove_pii)
    except RuntimeError as exc:
        raise RuntimeError(str(exc)) from exc
    jd_proc, resumes_proc = processed[0], processed[1:]

    candidates, meta = rank_resumes(
        jd_proc, resumes_proc, names,
        max_features=max_features, top_gap_terms=top_gaps,
    )
    meta["total_pipeline_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
    return candidates, meta, failures, scanned


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR  (rendered first so the appearance selector updates the theme)
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
<div class="sb-logo">
  <div class="sb-logo-mark">📄</div>
  <div>
    <div class="sb-logo-text-main">Smart Ranker</div>
    <div class="sb-logo-text-sub">Resume Intelligence</div>
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown('<div class="sb-cap">Appearance</div>', unsafe_allow_html=True)
    _sel = st.segmented_control(
        "Theme", ["Light", "Auto", "Dark"],
        default=st.session_state.get("theme_mode", "Auto"),
        format_func=lambda o: _THEME_LABELS.get(o, o),
        key="theme_seg", label_visibility="collapsed",
    )
    # The widget only returns a value once the user picks; keep the stored mode.
    if _sel is not None:
        st.session_state["theme_mode"] = _sel
    if st.session_state.get("_theme_fallback"):
        st.caption("⚠️ In-app switch blocked here — use ⋮ → Settings → Appearance.")

    st.divider()

    st.markdown('<div class="sb-cap">Analysis</div>', unsafe_allow_html=True)
    with st.container(border=True, key="side_settings"):
        remove_pii = st.checkbox("🔒 Redact PII (emails / phones / URLs)", value=True)
        st.markdown("<hr style='margin:.55rem 0;'>", unsafe_allow_html=True)
        max_features = st.slider(
            "Vocabulary size", 1_000, 20_000, 10_000, step=1_000,
            help="TF-IDF max features. 10 000 is the recommended default.",
        )
        top_gaps = st.slider(
            "Skill gaps per candidate", 5, 30, 15,
            help="Max JD keywords to surface as missing per resume.",
        )
        min_score = st.slider(
            "Min match threshold (%)", 0, 100, 0,
            help="Hide candidates below this score.",
        )

    st.divider()

    st.markdown('<div class="sb-cap">Privacy &amp; limits</div>', unsafe_allow_html=True)
    with st.container(border=True, key="side_privacy"):
        st.markdown(f"""
<div class="sb-info">
  <div>📎 <b>Formats:</b> {', '.join(s.upper().lstrip('.') for s in sorted(ALLOWED_EXTENSIONS))}</div>
  <div>📏 <b>Max size:</b> {MAX_FILE_SIZE_BYTES//1_048_576} MB per file</div>
  <div>📦 <b>Batch:</b> Up to {MAX_RESUMES} resumes</div>
  <div>🔐 <b>Storage:</b> In-memory only</div>
</div>""", unsafe_allow_html=True)

    st.divider()

    st.markdown("""
<div class="dev-card">
  <div class="dev-av">👨‍💻</div>
  <div class="dev-name">suman .builds</div>
  <div class="dev-role">Full-Stack Developer &amp; AI Engineer</div>
  <a class="dev-btn" href="https://suman-rana.netlify.app" target="_blank" rel="noopener">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
         stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
      <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>
      <polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/>
    </svg>
    View Portfolio
  </a>
</div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  APPLY THEME
# ═══════════════════════════════════════════════════════════════════════════════
_ACTIVE = resolve_base()
st.markdown(f"<style>{_css_for(_ACTIVE)}</style>", unsafe_allow_html=True)
apply_native_theme(_ACTIVE)


# ═══════════════════════════════════════════════════════════════════════════════
#  HERO
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <div class="hero-inner">
    <div class="hero-badge"><span class="pulse"></span> AI-Powered Resume Intelligence</div>
    <h1>Screen &amp; Rank Resumes<br><em>in Seconds</em></h1>
    <p>Upload a job description and up to 50 candidate resumes. Get explainable
    match scores, a ranked shortlist and per-candidate skill-gap analysis —
    all in under a second, nothing stored.</p>
    <div class="hero-chips">
      <span class="hero-chip">⚡ &lt; 500 ms · 50 resumes</span>
      <span class="hero-chip">🔒 Zero storage</span>
      <span class="hero-chip">🛡️ PII redaction</span>
      <span class="hero-chip">📊 TF-IDF · cosine</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

try:
    _load_nlp_cached()
except RuntimeError as exc:
    st.error(f"⚠️ {exc}")
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
#  INPUTS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-cap">Step 01 — Inputs</div>', unsafe_allow_html=True)
st.markdown('<div class="sec-title">Provide the role &amp; the candidates</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sec-sub">Paste or upload a Job Description on the left, '
    'then upload candidate resumes on the right. Both are required to run.</div>',
    unsafe_allow_html=True,
)

col_jd, col_res = st.columns(2, gap="large")

with col_jd:
    with st.container(border=True, key="card_jd"):
        st.markdown("""
<div class="uc-head">
  <div class="uc-icon uc-violet">📋</div>
  Job Description
</div>""", unsafe_allow_html=True)
        st.caption("Upload PDF / DOCX / TXT — or paste the text below.")
        jd_file = st.file_uploader(
            "Upload JD", type=["pdf", "docx", "txt"],
            key="jd", label_visibility="collapsed",
        )
        jd_pasted = st.text_area(
            "Or paste the JD here",
            height=175, placeholder="Paste the full job description here…",
        )

with col_res:
    with st.container(border=True, key="card_res"):
        st.markdown(f"""
<div class="uc-head">
  <div class="uc-icon uc-sky">📁</div>
  Candidate Resumes
  <span class="uc-hint">&nbsp;(up to {MAX_RESUMES})</span>
</div>""", unsafe_allow_html=True)
        st.caption("PDF or DOCX only · processed in memory, never written to disk.")
        resume_files = st.file_uploader(
            "Upload resumes", type=["pdf", "docx"],
            accept_multiple_files=True,
            key="resumes", label_visibility="collapsed",
        )
        if resume_files:
            if len(resume_files) > MAX_RESUMES:
                st.warning(f"Only the first {MAX_RESUMES} files will be processed.")
                resume_files = resume_files[:MAX_RESUMES]
            st.success(
                f"✅ {len(resume_files)} resume{'s' if len(resume_files) > 1 else ''} ready."
            )
        else:
            st.info("⬆️ Upload at least one resume to enable ranking.")

st.markdown("<div style='height:.9rem'></div>", unsafe_allow_html=True)
cta_l, cta_c, cta_r = st.columns([1, 2, 1])
with cta_c:
    analyze = st.button("🚀  Rank Candidates", type="primary", width="stretch")


# ═══════════════════════════════════════════════════════════════════════════════
#  ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
if analyze:
    jd_text = ""
    if jd_file is not None:
        try:
            raw = jd_file.getvalue()
            fname = sanitize_filename(jd_file.name)
            if fname.lower().endswith(".txt"):
                try:
                    jd_text = raw.decode("utf-8", errors="ignore")
                except Exception as exc:
                    st.error(f"Could not decode JD text file: {exc}"); st.stop()
                finally:
                    del raw
                if len(jd_text.strip()) < 10:
                    st.error("JD text file appears empty."); st.stop()
            else:
                try:
                    jd_text = extract_text(raw, fname)
                except ScannedDocumentError as exc:
                    st.warning(f"JD appears scanned/image-only: {exc}"); st.stop()
                except EncryptedDocumentError as exc:
                    st.error(f"JD is encrypted: {exc}"); st.stop()
                except (FileValidationError, TextExtractionError) as exc:
                    st.error(f"Could not read JD: {exc}"); st.stop()
                finally:
                    del raw
        except Exception as exc:
            st.error(f"Could not read JD: {exc}"); st.stop()

    if not jd_text.strip() and jd_pasted.strip():
        jd_text = jd_pasted.strip()
    if not jd_text.strip():
        st.error("Please provide a Job Description before ranking."); st.stop()
    if len(jd_text.strip()) < 10:
        st.error("Job Description is too short — please provide more detail."); st.stop()
    if not resume_files:
        st.error("Please upload at least one resume."); st.stop()

    with st.spinner("⚙️ Extracting · preprocessing · scoring…"):
        try:
            candidates, meta, failures, scanned = _process_batch(
                jd_text, list(resume_files), remove_pii, max_features, top_gaps,
            )
        except ValueError as exc:
            st.error(str(exc)); st.stop()
        except RuntimeError as exc:
            st.error(str(exc)); st.stop()
        except Exception as exc:
            logger.exception("Batch failed")
            st.error(f"Ranking failed unexpectedly: {exc}"); st.stop()

    # Persist so later reruns (theme switch, slider tweaks, tab switch) keep the
    # shortlist on screen instead of dropping back to the empty state.
    st.session_state["_results"] = (candidates, meta, failures, scanned)


_results_view = st.session_state.get("_results")
if _results_view is not None:
    candidates, meta, failures, scanned = _results_view

    for fn, err in failures:
        st.error(f"❌ Skipped **{sanitize_filename(fn)}**: {err}")
    for fn, note in scanned:
        st.warning(f"📷 **{sanitize_filename(fn)}** is scanned/image-only — skipped. {note}")

    if candidates and all(c.score == 0.0 for c in candidates):
        st.warning(
            "⚠️ Zero vocabulary overlap — all scores are 0.0 %. "
            "Try a more detailed JD or check resumes contain selectable text."
        )

    visible = [c for c in candidates if c.score >= min_score]
    if not visible:
        st.warning("No candidates meet the minimum score. Try lowering the threshold.")
        st.stop()

    # ── Results header ────────────────────────────────────────────────────────
    hdr_l, hdr_r = st.columns([4, 1], vertical_alignment="center")
    with hdr_l:
        st.markdown('<div class="sec-cap">Step 02 — Results</div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">Ranked shortlist</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sec-sub">Scores are TF-IDF cosine similarity (0–100 %). '
            'Higher = stronger keyword alignment with the JD.</div>',
            unsafe_allow_html=True,
        )
    with hdr_r:
        if st.button("↺  New analysis", width="stretch"):
            st.session_state.pop("_results", None)
            st.rerun()

    # ── KPI row ───────────────────────────────────────────────────────────────
    scores = [c.score for c in visible]
    avg_s = sum(scores) / len(scores)
    k1, k2, k3, k4 = st.columns(4, gap="medium")
    for col, key, ico, lbl, val, sub in [
        (k1, "kpi_1", "👥", "Candidates", str(len(visible)), "ranked"),
        (k2, "kpi_2", "🥇", "Top score", f"{max(scores):.1f}%", visible[0].name),
        (k3, "kpi_3", "📈", "Average", f"{avg_s:.1f}%", "across shortlist"),
        (k4, "kpi_4", "⚡", "Pipeline", f"{meta.get('total_pipeline_ms',0):.0f}",
         f"ms · {int(meta.get('vocab_size',0)):,} tokens"),
    ]:
        with col:
            with st.container(border=True, key=key):
                st.markdown(
                    f'<div class="kpi">'
                    f'<div class="kpi-ico">{ico}</div>'
                    f'<div class="kpi-lbl">{lbl}</div>'
                    f'<div class="kpi-val">{val}</div>'
                    f'<div class="kpi-sub">{sub}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # ── DataFrame ─────────────────────────────────────────────────────────────
    medals = {1: "🥇", 2: "🥈", 3: "🥉"}
    df = pd.DataFrame([
        {
            "": medals.get(c.rank, f"#{c.rank}"),
            "Rank": c.rank,
            "Candidate": c.name,
            "Match %": c.score,
            "Key gaps": ", ".join(c.skill_gaps[:8]) or "—",
        }
        for c in visible
    ])

    # ── Spotlight ─────────────────────────────────────────────────────────────
    top = visible[0]
    gap_txt = (
        f"Key gaps: {', '.join(top.skill_gaps[:5])}"
        if top.skill_gaps else "Excellent coverage — no major gaps."
    )
    st.markdown(f"""
<div class="spotlight">
  <div class="sp-left">
    <div class="sp-eyebrow">Top match</div>
    <div class="sp-name">#{top.rank} &nbsp; {top.name}</div>
    <div class="sp-sub">
      Strongest keyword alignment with the job description.<br>{gap_txt}
    </div>
  </div>
  <div class="sp-right">
    <div class="sp-score">{top.score:.1f}%</div>
    {_badge(top.score)}
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_rank, tab_dist, tab_gaps = st.tabs(
        ["🏆  Rankings", "📊  Distribution", "🔍  Skill gaps"]
    )

    with tab_rank:
        st.markdown("<div style='height:.35rem'></div>", unsafe_allow_html=True)
        st.dataframe(
            df, width="stretch", hide_index=True,
            column_config={
                "Match %": st.column_config.ProgressColumn(
                    "Match %", format="%.1f%%", min_value=0, max_value=100,
                ),
            },
        )
        st.markdown("**Match score by candidate**")
        st.bar_chart(df.set_index("Candidate")["Match %"], height=320, color="#7C3AED")

    with tab_dist:
        st.markdown("<div style='height:.35rem'></div>", unsafe_allow_html=True)
        try:
            hist_df = pd.cut(pd.Series(scores), bins=10, include_lowest=True)
            dist = hist_df.value_counts().sort_index()
            dist.index = [f"{iv.left:.0f}–{iv.right:.0f}%" for iv in dist.index]
            st.bar_chart(dist, height=320, color="#7C3AED")
            st.caption(
                f"Score distribution across {len(visible)} candidates (10 bins). "
                "Each bar = number of candidates in that score range."
            )
        except Exception as exc:
            logger.warning("Distribution chart failed: %s", exc)
            st.warning("Could not render score distribution.")

    with tab_gaps:
        st.markdown("<div style='height:.35rem'></div>", unsafe_allow_html=True)
        for c in visible:
            med = medals.get(c.rank, f"#{c.rank}")
            with st.expander(f"{med}  {c.name}  —  {c.score:.1f}%", expanded=(c.rank == 1)):
                p_col, b_col = st.columns([5, 1])
                with p_col:
                    st.progress(min(max(c.score / 100.0, 0.0), 1.0))
                with b_col:
                    st.markdown(_badge(c.score), unsafe_allow_html=True)
                if c.skill_gaps:
                    st.markdown(
                        "<div style='margin:.5rem 0 .25rem; font-size:.82rem;"
                        "font-weight:600; color:var(--tx-3);'>"
                        "Missing / weak JD keywords:</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        " ".join(f'<span class="tag">{s}</span>' for s in c.skill_gaps),
                        unsafe_allow_html=True,
                    )
                else:
                    st.success("✅ Strong keyword coverage — no major gaps detected.")

    # ── Export ────────────────────────────────────────────────────────────────
    st.divider()
    ex_l, ex_c, ex_r = st.columns([1, 2, 1])
    with ex_c:
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        csv_str = buf.getvalue()
        buf.close()
        st.download_button(
            "⬇️  Download rankings (CSV)",
            data=csv_str, file_name="resume_rankings.csv",
            mime="text/csv", width="stretch",
        )
    with st.expander("ℹ️ Performance & privacy details"):
        st.json({k: v for k, v in meta.items()})
        st.caption(
            "Sparse TF-IDF (O(N·M)) · cosine similarity on CSR matrices · "
            "no files written to disk · PII optionally redacted before scoring."
        )

else:
    # ═════════════════════════════════════════════════════════════════════════
    #  EMPTY STATE — How it works
    # ═════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="sec-cap">How it works</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sec-title">Three steps to a ranked shortlist</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sec-sub">No account required · nothing stored · '
        'results in under a second.</div>',
        unsafe_allow_html=True,
    )

    s1, s2, s3 = st.columns(3, gap="medium")
    for col, key, num, icon, title, body in [
        (s1, "card_step1", "01", "📥", "Upload",
         "Drop in a Job Description (PDF, DOCX or TXT) and up to 50 candidate "
         "resumes. Everything is processed in memory — nothing touches disk."),
        (s2, "card_step2", "02", "🧠", "Clean &amp; match",
         "Optional PII redaction → spaCy lemmatization → TF-IDF vectorization → "
         "sparse cosine similarity. Full pipeline in under 500 ms for 50 resumes."),
        (s3, "card_step3", "03", "🏆", "Shortlist",
         "Ranked table with match percentages, interactive score distribution, "
         "per-candidate skill-gap tags, and one-click CSV export."),
    ]:
        with col:
            with st.container(border=True, key=key):
                st.markdown(
                    f'<div class="step">'
                    f'<div class="step-n">Step {num}</div>'
                    f'<span class="step-icon">{icon}</span>'
                    f'<div class="step-t">{title}</div>'
                    f'<div class="step-b">{body}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
    st.info(
        "👆 &nbsp; Fill in the **Job Description** and **Candidate Resumes** above, "
        "then click **🚀 Rank Candidates**."
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="footer">'
    "Smart Resume Ranker &nbsp;·&nbsp; Decision-support tool — keep a human in the loop"
    " &nbsp;·&nbsp; In-memory only, zero storage"
    " &nbsp;·&nbsp; Built by "
    "<a href='https://suman-rana.netlify.app' target='_blank' rel='noopener'>suman .builds</a>"
    "</div>",
    unsafe_allow_html=True,
)
