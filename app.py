"""Smart Resume Ranker — premium Streamlit dashboard.

Workflow:
  1. Upload JD (+ optional paste) and multiple resumes (PDF/DOCX).
  2. In-memory extraction (no disk writes) -> spaCy preprocessing.
  3. Sparse TF-IDF + cosine similarity ranking.
  4. Interactive rankings, distribution chart, skill gaps, CSV export.

Security: filenames sanitized, extension whitelist + magic-byte signature
check, 10 MB/file cap, PII redaction toggle, zero persistent storage.
Robustness: corrupt / encrypted / scanned files are isolated per-file
(``st.error`` / ``st.warning``) and never crash the batch; zero-overlap
yields 0.0 scores with an explanatory warning; empty inputs are rejected
up front with clear guidance.
"""

from __future__ import annotations

import io
import logging
import time
from typing import List, Tuple

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

# ---------------------------------------------------------------- page setup
st.set_page_config(
    page_title="Smart Resume Ranker",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

MAX_RESUMES = 50

# ------------------------------------------------------- premium stylesheet
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem; }

/* ---- Slim transparent header: keeps the sidebar collapse/expand toggle
        working, while all other chrome (toolbar / menu) is removed below ---- */
[data-testid="stHeader"],
header[data-testid="stHeader"],
header.stAppHeader { background: transparent !important; }

/* ---- Sidebar open/close controls must ALWAYS stay visible & clickable ---- */
[data-testid^="stSidebarCollapse"],
[data-testid^="stSidebarCollapsed"] {
  display: flex !important;
  visibility: visible !important;
  opacity: 1 !important;
  pointer-events: auto !important;
}

/* Hero banner */
.hero {
  background:
    radial-gradient(1200px 300px at 85% -10%, rgba(6,182,212,.35), transparent 60%),
    linear-gradient(135deg, #0B1B33 0%, #1D4ED8 58%, #06B6D4 100%);
  border: 1px solid rgba(255,255,255,.14);
  border-radius: 20px; padding: 2.2rem 2.4rem; color: #fff;
  box-shadow: 0 20px 45px rgba(29,78,216,.30); margin-bottom: 1.5rem;
}
.hero h1 { color: #fff !important; font-size: 2rem; font-weight: 800; margin: 0; letter-spacing: -.02em; }
.hero p { color: rgba(255,255,255,.85); margin: .5rem 0 1rem; font-size: 1rem; }
.hero .pills { display: flex; gap: .5rem; flex-wrap: wrap; }
.hero .pill {
  background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.25);
  color: #fff; font-size: .78rem; font-weight: 600; padding: .3rem .75rem; border-radius: 999px;
}

/* Section titles */
.section-label {
  font-size: .75rem; font-weight: 700; letter-spacing: .12em;
  color: #0284C7; text-transform: uppercase; margin: 1.6rem 0 .4rem;
}
.section-title { font-size: 1.25rem; font-weight: 700; color: #0F172A; margin: 0 0 .2rem; }
.section-sub { color: #64748B; font-size: .9rem; margin-bottom: 1rem; }

/* Cards (applied to bordered containers) */
[data-testid="stVerticalBlockBorderWrapper"] {
  border-radius: 16px !important;
  box-shadow: 0 8px 24px rgba(15,23,42,.06);
}

/* KPI cards */
.kpi { text-align: center; padding: .4rem .2rem; }
.kpi .kpi-label { font-size: .72rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: #64748B; }
.kpi .kpi-value { font-size: 1.7rem; font-weight: 800; color: #0F172A; line-height: 1.2; }
.kpi .kpi-sub { font-size: .78rem; color: #64748B; }

/* Top-candidate spotlight */
.spotlight {
  background:
    radial-gradient(600px 160px at 90% 0%, rgba(6,182,212,.4), transparent 60%),
    linear-gradient(135deg, #0B1B33, #1D4ED8);
  border: 1px solid rgba(255,255,255,.14);
  border-radius: 16px; padding: 1.2rem 1.5rem; color: #fff;
  box-shadow: 0 12px 30px rgba(29,78,216,.35); margin: 1rem 0;
}
.spotlight .spot-label { font-size: .72rem; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; color: rgba(255,255,255,.75); }
.spotlight .spot-name { font-size: 1.3rem; font-weight: 800; margin: .15rem 0; }
.spotlight .spot-sub { font-size: .85rem; color: rgba(255,255,255,.85); }

/* Score badges */
.badge { display: inline-block; font-weight: 700; font-size: .8rem; padding: .2rem .65rem; border-radius: 999px; }
.badge-green { background: #DCFCE7; color: #166534; }
.badge-indigo { background: #E0E7FF; color: #3730A3; }
.badge-amber { background: #FEF3C7; color: #92400E; }
.badge-red { background: #FEE2E2; color: #991B1B; }

/* Skill tags */
.skill-tag {
  display: inline-block; background: #E0F2FE; color: #0C4A6E; border: 1px solid #7DD3FC;
  font-size: .76rem; font-weight: 600; padding: .18rem .6rem; border-radius: 999px; margin: .12rem;
}

/* CTA button */
div.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, #1D4ED8, #06B6D4);
  border: none; border-radius: 12px; padding: .75rem 1rem;
  font-weight: 700; font-size: 1rem; box-shadow: 0 10px 25px rgba(29,78,216,.35);
}
div.stButton > button[kind="primary"]:hover { filter: brightness(1.07); }
div.stDownloadButton > button {
  border-radius: 12px; font-weight: 700; border: 1.5px solid #1D4ED8; color: #1D4ED8;
}

/* Uploaders: equal, tidy */
[data-testid="stFileUploader"] { border-radius: 12px; }
[data-testid="stFileUploaderDropzone"] { min-height: 120px; }

/* ---- Remove Streamlit chrome: top-right toolbar, main menu, footer ---- */
[data-testid="stAppToolbar"],
div[class*="stAppToolbar"] { display: none !important; }
#MainMenu { visibility: hidden !important; }
footer { visibility: hidden !important; }
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; }

/* Center helpers */
.center { text-align: center; }
.footer { text-align: center; color: #94A3B8; font-size: .78rem; padding: 1.5rem 0 .5rem; }
</style>
    """,
    unsafe_allow_html=True,
)


def _score_badge(score: float) -> str:
    """Return a coloured HTML badge for a 0-100 score."""
    if score >= 75:
        cls = "badge-green"
    elif score >= 50:
        cls = "badge-indigo"
    elif score >= 25:
        cls = "badge-amber"
    else:
        cls = "badge-red"
    return f'<span class="badge {cls}">{score:.1f}%</span>'


@st.cache_resource(show_spinner="Loading language model…")
def _load_nlp_cached() -> object:
    return get_nlp()


def _read_upload(uploaded_file: object) -> Tuple[str, bytes]:
    """Read a Streamlit UploadedFile into memory with size guard."""
    name = sanitize_filename(getattr(uploaded_file, "name", "unnamed"))
    try:
        data: bytes = uploaded_file.getvalue()  # type: ignore[attr-defined,union-attr]
    except Exception as exc:
        raise ValueError(f"Could not read '{name}': {exc}") from exc
    if len(data) > MAX_FILE_SIZE_BYTES:
        raise ValueError(
            f"'{name}' is {len(data) / 1_048_576:.1f} MB "
            f"(limit {MAX_FILE_SIZE_BYTES // 1_048_576} MB)."
        )
    if not data:
        raise ValueError(f"'{name}' is empty.")
    return name, data


def _process_batch(
    jd_text: str, resume_files: list, remove_pii: bool, max_features: int, top_gaps: int
) -> Tuple[List[RankedCandidate], dict, List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Extract -> preprocess -> rank.

    Returns:
        (candidates, meta, failures, scanned_notes) where failures are
        hard errors (corrupt/encrypted/invalid) and scanned_notes are
        image-only docs reported separately as warnings.
    """
    t_start = time.perf_counter()
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
                scanned.append((orig, str(exc)))
                logger.warning("Scanned document skipped: %s", orig)
                continue
            except EncryptedDocumentError as exc:
                failures.append((orig, f"Encrypted: {exc}"))
                logger.warning("Encrypted document skipped: %s", orig)
                continue
            except (FileValidationError, TextExtractionError) as exc:
                failures.append((orig, str(exc)))
                logger.warning("Extraction failed for %s: %s", orig, exc)
                continue
            finally:
                del data  # free bytes promptly
            if len(text.strip()) < 10:
                scanned.append(
                    (orig, "Almost no selectable text found — likely scanned images.")
                )
                continue
            names.append(safe_name)
            raw_texts.append(text)
        except ValueError as exc:  # size / empty / read errors
            failures.append((orig, str(exc)))
        except Exception as exc:  # per-file isolation: one bad file ≠ total failure
            logger.exception("Unexpected error for %s", orig)
            failures.append((orig, f"Unexpected error: {exc}"))

    if not names:
        raise ValueError(
            "No valid resumes could be extracted. "
            "Files may be scanned images, encrypted, or corrupted — see warnings above."
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
    meta["total_pipeline_ms"] = round((time.perf_counter() - t_start) * 1000.0, 2)
    return candidates, meta, failures, scanned


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### 📄 Smart Ranker")
    st.caption("Objective resume screening · TF-IDF + cosine similarity")
    theme = st.segmented_control(
        "Appearance",
        options=["Light", "Auto", "Dark"],
        default="Auto",
        key="theme_mode",
        help="Light / Dark switch Streamlit's native theme (every widget, "
        "table and chart re-renders perfectly). Auto follows your OS setting.",
    )
    if st.session_state.get("_theme_fallback"):
        st.caption("⚠️ In-app switch blocked here — use ⋮ → Settings → Appearance.")
    st.divider()
    with st.container(border=True):
        st.markdown("**⚙️ Analysis settings**")
        remove_pii = st.checkbox("Redact PII (emails / phones / URLs)", value=True)
        max_features = st.slider("TF-IDF max features", 1_000, 20_000, 10_000, step=1_000)
        top_gaps = st.slider("Skill gaps per candidate", 5, 30, 15)
        min_score = st.slider("Minimum score filter (%)", 0, 100, 0)
    with st.container(border=True):
        st.markdown("**🔒 Privacy & limits**")
        st.caption(
            f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}  \n"
            f"Max {MAX_FILE_SIZE_BYTES // 1_048_576} MB / file · {MAX_RESUMES} resumes max  \n"
            "In-memory only — nothing is stored."
        )

# ------------------------------------------------- native theme engine
# Streamlit renders tables, charts, alerts and widgets itself, so any
# hand-written "dark mode" CSS always leaves mismatched light widgets
# behind. Instead we switch Streamlit's NATIVE base theme (all widgets
# re-render perfectly) and only adapt our own small brand classes.
# NOTE: set_option mutates process-global config — fine for this local
# single-user app; on a multi-user server prefer per-user Settings menu.
import streamlit.components.v1 as components

if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "Auto"


def _resolve_theme_mode() -> str:
    """Resolve Light/Auto/Dark to a concrete Light/Dark choice."""
    mode = st.session_state.get("theme_mode", "Auto")
    if mode != "Auto":
        return mode
    os_theme = st.query_params.get("os_theme", "")
    if os_theme in ("light", "dark"):
        return "Dark" if os_theme == "dark" else "Light"
    return "Light"  # before the one-time OS detector below reports back


def _apply_native_theme(mode: str) -> None:
    """Apply the native base theme; reruns once only when it changes."""
    want = "dark" if mode == "Dark" else "light"
    try:
        current = st.get_option("theme.base") or "light"
        if current != want:
            st._config.set_option("theme.base", want)  # noqa: SLF001
            st.rerun()
    except Exception as exc:
        logger.warning("Native theme switch unavailable: %s", exc)
        st.session_state["_theme_fallback"] = True


# One-time OS detection for Auto mode: runs only on fresh loads without
# uploads (sets ?os_theme= then reloads once; never loops).
if st.session_state.theme_mode == "Auto" and "os_theme" not in st.query_params:
    components.html(
        """<script>
try {
  var dark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  var url = new URL(window.top.location.href);
  if (!url.searchParams.get('os_theme')) {
    url.searchParams.set('os_theme', dark ? 'dark' : 'light');
    window.top.location.replace(url.toString());
  }
} catch (e) {}
</script>""",
        height=0,
    )

_apply_native_theme(_resolve_theme_mode())


def _active_theme_type() -> str:
    """The theme Streamlit is actually rendering ('light'/'dark')."""
    try:
        t = st.context.theme.get("type")
        return t if t in ("light", "dark") else "light"
    except Exception:
        return "light"


# Adapt ONLY our own brand classes to the active native theme.
# (Native widgets/tables/charts/alerts already match — nothing to fix.)
if _active_theme_type() == "dark":
    st.markdown(
        """<style>
.section-title { color: #F1F5F9; }
.section-sub { color: #94A3B8; }
.kpi .kpi-value { color: #F8FAFC; }
.kpi .kpi-label, .kpi .kpi-sub { color: #94A3B8; }
.skill-tag { background: #1E2A5A; color: #BAE6FD; border-color: #0284C7; }
.footer { color: #64748B; }
</style>""",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------- hero (centered, aligned)
st.markdown(
    """
<div class="hero">
  <h1>Smart Resume Screening &amp; Ranking</h1>
  <p>Upload a job description and candidate resumes — get explainable match scores,
  ranked shortlists and per-candidate skill gaps in seconds.</p>
  <div class="pills">
    <span class="pill">In-memory · zero storage</span>
    <span class="pill">PII redaction built-in</span>
    <span class="pill">50+ resumes &lt; 500 ms scoring</span>
  </div>
</div>
    """,
    unsafe_allow_html=True,
)

try:
    _load_nlp_cached()
except RuntimeError as exc:
    st.error(f"⚠️ {exc}")
    st.stop()

# ------------------------------------------------- inputs (aligned cards)
st.markdown('<div class="section-label">Step 01 — Inputs</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-title">Provide the role and the candidates</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="section-sub">Both panels are aligned — complete each side, then run the ranking.</div>',
    unsafe_allow_html=True,
)

col_jd, col_res = st.columns(2, gap="large")

with col_jd:
    with st.container(border=True):
        st.markdown("**1️⃣ &nbsp; Job description**")
        st.caption("Upload a file or paste the text — either one is enough.")
        jd_file = st.file_uploader(
            "Upload JD (PDF / DOCX / TXT)",
            type=["pdf", "docx", "txt"],
            key="jd",
        )
        jd_pasted = st.text_area(
            "Or paste JD text",
            height=180,
            placeholder="Paste the job description here…",
        )

with col_res:
    with st.container(border=True):
        st.markdown(f"**2️⃣ &nbsp; Candidate resumes** <span style='color:#64748B'>(up to {MAX_RESUMES})</span>")
        st.caption("PDF or DOCX · each file processed in memory, in isolation.")
        resume_files = st.file_uploader(
            "Upload resumes",
            type=["pdf", "docx"],
            accept_multiple_files=True,
            key="resumes",
        )
        # Fixed-height status strip keeps both cards visually aligned.
        if resume_files:
            if len(resume_files) > MAX_RESUMES:
                st.warning(f"Only the first {MAX_RESUMES} files will be processed.")
                resume_files = resume_files[:MAX_RESUMES]
            st.success(f"✅ {len(resume_files)} resume(s) ready for ranking.")
        else:
            st.info("No resumes yet — upload at least one to enable ranking.")

# Centered CTA
cta_l, cta_c, cta_r = st.columns([1, 2, 1])
with cta_c:
    analyze = st.button("🚀 Rank Candidates", type="primary", use_container_width=True)

if analyze:
    # ---- empty-input validation (JD + at least one resume required)
    jd_text = ""
    if jd_file is not None:
        try:
            raw = jd_file.getvalue()
            fname = sanitize_filename(jd_file.name)
            if fname.lower().endswith(".txt"):
                try:
                    jd_text = raw.decode("utf-8", errors="ignore")
                except Exception as exc:
                    st.error(f"Could not decode JD text file: {exc}")
                    st.stop()
                if len(jd_text.strip()) < 10:
                    st.error("JD text file appears empty or image-only.")
                    st.stop()
            else:
                try:
                    jd_text = extract_text(raw, fname)
                except ScannedDocumentError as exc:
                    st.warning(f"JD looks scanned/image-only: {exc}")
                    st.stop()
                except EncryptedDocumentError as exc:
                    st.error(f"JD is encrypted: {exc}")
                    st.stop()
                except (FileValidationError, TextExtractionError) as exc:
                    st.error(f"Could not read Job Description: {exc}")
                    st.stop()
        except Exception as exc:
            st.error(f"Could not read Job Description: {exc}")
            st.stop()
    if not jd_text.strip() and jd_pasted.strip():
        jd_text = jd_pasted.strip()
    if not jd_text.strip():
        st.error("Please upload or paste a Job Description before ranking.")
        st.stop()
    if len(jd_text.strip()) < 10:
        st.error("Job Description is too short — please provide more detail.")
        st.stop()
    if not resume_files:
        st.error("Please upload at least one resume before ranking.")
        st.stop()

    with st.spinner("Extracting, preprocessing & scoring…"):
        try:
            candidates, meta, failures, scanned = _process_batch(
                jd_text, list(resume_files), remove_pii, max_features, top_gaps
            )
        except ValueError as exc:
            st.error(str(exc))
            st.stop()
        except RuntimeError as exc:
            st.error(str(exc))
            st.stop()
        except Exception as exc:  # defensive: never show a raw traceback
            logger.exception("Batch failed")
            st.error(f"Ranking failed unexpectedly: {exc}")
            st.stop()

    # ---- per-file graceful-degradation notices
    for fname, err in failures:
        st.error(f"❌ Skipped **{sanitize_filename(fname)}**: {err}")
    for fname, note in scanned:
        st.warning(
            f"📷 **{sanitize_filename(fname)}** appears to be a scanned / image-only "
            f"document and was skipped. {note}"
        )

    # ---- zero-overlap guard
    if candidates and all(c.score == 0.0 for c in candidates):
        st.warning(
            "⚠️ Zero vocabulary overlap: the JD and resumes share no common "
            "keywords after preprocessing, so all scores are 0.0%. Try a more "
            "detailed JD or check that resumes contain selectable text."
        )

    # ---- score filter
    visible = [c for c in candidates if c.score >= min_score]
    if not visible:
        st.warning("No candidates meet the minimum score filter.")
        st.stop()

    # ------------------------------------------------ results header
    st.markdown('<div class="section-label">Step 02 — Results</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Ranked shortlist</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Scores are TF-IDF cosine similarity (0–100%). '
        "Higher means stronger keyword alignment with the JD.</div>",
        unsafe_allow_html=True,
    )

    # ---- KPI cards (aligned, centered)
    scores = [c.score for c in visible]
    avg_score = sum(scores) / len(scores)
    k1, k2, k3, k4 = st.columns(4, gap="medium")
    with k1:
        with st.container(border=True):
            st.markdown(
                f'<div class="kpi"><div class="kpi-label">Candidates</div>'
                f'<div class="kpi-value">{len(visible)}</div>'
                f'<div class="kpi-sub">ranked</div></div>',
                unsafe_allow_html=True,
            )
    with k2:
        with st.container(border=True):
            st.markdown(
                f'<div class="kpi"><div class="kpi-label">Top score</div>'
                f'<div class="kpi-value">{max(scores):.1f}%</div>'
                f'<div class="kpi-sub">{visible[0].name}</div></div>',
                unsafe_allow_html=True,
            )
    with k3:
        with st.container(border=True):
            st.markdown(
                f'<div class="kpi"><div class="kpi-label">Average</div>'
                f'<div class="kpi-value">{avg_score:.1f}%</div>'
                f'<div class="kpi-sub">across shortlist</div></div>',
                unsafe_allow_html=True,
            )
    with k4:
        with st.container(border=True):
            st.markdown(
                f'<div class="kpi"><div class="kpi-label">Pipeline</div>'
                f'<div class="kpi-value">{meta.get("total_pipeline_ms", 0):.0f}<span style="font-size:.9rem"> ms</span></div>'
                f'<div class="kpi-sub">vocab {int(meta.get("vocab_size", 0))}</div></div>',
                unsafe_allow_html=True,
            )

    df = pd.DataFrame(
        [
            {
                "": {1: "🥇", 2: "🥈", 3: "🥉"}.get(c.rank, "—"),
                "Rank": c.rank,
                "Candidate": c.name,
                "Match %": c.score,
                "Top missing skills": ", ".join(c.skill_gaps[:8]) or "—",
            }
            for c in visible
        ]
    )

    # ---- top-candidate spotlight (strongest signal, first thing seen)
    top = visible[0]
    st.markdown(
        f"""
<div class="spotlight">
  <div class="spot-label">⭐ Top match</div>
  <div class="spot-name">#{top.rank} {top.name} — {top.score:.1f}%</div>
  <div class="spot-sub">Strongest keyword alignment with the JD.
  {("Key gaps: " + ", ".join(top.skill_gaps[:5])) if top.skill_gaps else "Excellent coverage — no major gaps."}</div>
</div>
        """,
        unsafe_allow_html=True,
    )

    # ---- tabs keep tables / charts / gaps neatly aligned
    tab_rank, tab_dist, tab_gaps = st.tabs(
        ["🏆 Rankings", "📊 Score distribution", "🔍 Skill gaps"]
    )

    with tab_rank:
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.markdown("**Match score by candidate**")
        st.bar_chart(df.set_index("Candidate")["Match %"])

    with tab_dist:
        try:
            hist_df = pd.cut(pd.Series(scores), bins=10, include_lowest=True)
            dist = hist_df.value_counts().sort_index()
            dist.index = [f"{iv.left:.0f}–{iv.right:.0f}%" for iv in dist.index]
            st.bar_chart(dist)
            st.caption("Distribution of match scores across the shortlist (10 bins).")
        except Exception as exc:
            logger.warning("Distribution chart failed: %s", exc)
            st.warning("Could not render score distribution.")

    with tab_gaps:
        for c in visible:
            with st.expander(f"#{c.rank}  {c.name}  —  {c.score:.1f}%", expanded=(c.rank == 1)):
                st.progress(min(max(c.score / 100.0, 0.0), 1.0))
                st.markdown(_score_badge(c.score), unsafe_allow_html=True)
                if c.skill_gaps:
                    st.markdown("**Missing / weak JD keywords:**")
                    st.markdown(
                        " ".join(f'<span class="skill-tag">{s}</span>' for s in c.skill_gaps),
                        unsafe_allow_html=True,
                    )
                else:
                    st.success("Strong keyword coverage — no major gaps detected.")

    # ---- export + details (centered, aligned)
    st.divider()
    e_l, e_c, e_r = st.columns([1, 2, 1])
    with e_c:
        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        st.download_button(
            "⬇️ Download rankings (CSV)",
            data=csv_buf.getvalue(),
            file_name="resume_rankings.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with st.expander("ℹ️ Performance & privacy details"):
        st.json({k: v for k, v in meta.items()})
        st.caption(
            "Time O(N·M) via sparse TF-IDF; space O(nnz) — matrices never densified. "
            "Files processed in memory (BytesIO) and released immediately; "
            "no resumes or PII written to disk."
        )
else:
    # --------------------------------------- premium empty state (aligned)
    st.markdown('<div class="section-label">How it works</div>', unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3, gap="medium")
    with s1:
        with st.container(border=True):
            st.markdown("**📥 1 · Upload**")
            st.caption("JD + up to 50 resumes (PDF/DOCX). Files stay in memory.")
    with s2:
        with st.container(border=True):
            st.markdown("**🧹 2 · Clean & match**")
            st.caption("PII redaction → spaCy lemmas → TF-IDF cosine similarity.")
    with s3:
        with st.container(border=True):
            st.markdown("**🏆 3 · Shortlist**")
            st.caption("Ranked table, charts, skill gaps + one-click CSV export.")
    st.info("👆 Upload a JD and resumes above, then click **Rank Candidates**.")

st.markdown(
    '<div class="footer">Smart Resume Ranker · decision support — keep a human in the loop · '
    "in-memory only, nothing stored</div>",
    unsafe_allow_html=True,
)
