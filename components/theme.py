"""Theme component for Smart Resume Ranker."""

from __future__ import annotations

import streamlit as st


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
    """Apply the native base theme; reruns once only when it changes."""
    want = "dark" if mode == "Dark" else "light"
    try:
        current = st.get_option("theme.base") or "light"
        if current != want:
            st._config.set_option("theme.base", want)  # noqa: SLF001
            st.rerun()
    except Exception as exc:
        st.session_state["_theme_fallback"] = True
        st.warning(f"Native theme switch unavailable: {exc}")


def adapt_brand_classes() -> None:
    """Adapt ONLY our own brand classes to the active native theme."""
    try:
        active = st.context.theme.get("type")
        theme_type = active if active in ("light", "dark") else "light"
    except Exception:
        theme_type = "light"

    if theme_type == "dark":
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