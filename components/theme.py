"""Theme component for Smart Resume Ranker."""

from __future__ import annotations

import logging

import streamlit as st

logger = logging.getLogger(__name__)

_PALETTES = {
    "light": {
        "theme.base": "light",
        "theme.primaryColor": "#1D4ED8",
        "theme.backgroundColor": "#F7F8FC",
        "theme.secondaryBackgroundColor": "#FFFFFF",
        "theme.textColor": "#0F172A",
    },
    "dark": {
        "theme.base": "dark",
        "theme.primaryColor": "#38BDF8",
        "theme.backgroundColor": "#0B1120",
        "theme.secondaryBackgroundColor": "#131A33",
        "theme.textColor": "#E2E8F0",
    },
}


def resolve_theme_mode() -> str:
    """Resolve Light/Auto/Dark to a concrete Light/Dark choice."""
    mode = st.session_state.get("theme_mode", "Auto")
    if mode != "Auto":
        return mode
    os_theme = st.query_params.get("os_theme", "")
    if os_theme in ("light", "dark"):
        return "Dark" if os_theme == "dark" else "Light"
    return "Light"


def apply_native_theme(mode: str) -> None:
    """Apply the native base theme with full palette; reruns once only when it changes."""
    want = "dark" if mode == "Dark" else "light"
    # Guard: skip if we already applied this base in a previous run.
    if st.session_state.get("_applied_base") == want:
        return
    try:
        pal = _PALETTES[want]
        needs_update = False
        for k, v in pal.items():
            try:
                cur = st.get_option(k)
            except Exception:
                cur = None
            if cur != v:
                needs_update = True
                break
        if needs_update:
            for k, v in pal.items():
                try:
                    st._config.set_option(k, v)  # noqa: SLF001
                except Exception:
                    logger.warning("Failed setting theme option %s", k)
            st.session_state["_applied_base"] = want
            st.rerun()
        else:
            st.session_state["_applied_base"] = want
    except Exception as exc:
        logger.warning("Native theme switch unavailable: %s", exc)
        st.session_state["_theme_fallback"] = True


def adapt_brand_classes(active_base: str) -> None:
    """Adapt ONLY our own brand classes to the active native theme.

    Args:
        active_base: The resolved base ('light' or 'dark').
    """
    if active_base == "dark":
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