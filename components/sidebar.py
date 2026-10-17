"""Sidebar component for Smart Resume Ranker."""
from __future__ import annotations

import streamlit as st

from config import MAX_RESUMES
from utils.parser import ALLOWED_EXTENSIONS, MAX_FILE_SIZE_BYTES


def render_sidebar() -> tuple[bool, int, int, int]:
    """Render the application sidebar with theme and settings.

    Returns:
        (remove_pii, max_features, top_gaps, min_score) — the four
        analysis-settings widget values that the main app needs to pass
        to the processing pipeline.
    """
    with st.sidebar:
        st.markdown("### 📄 Smart Ranker")
        st.caption("Objective resume screening · TF-IDF + cosine similarity")
        st.segmented_control(
            "Appearance",
            options=["Light", "Auto", "Dark"],
            default="Auto",
            key="theme_mode",
            help=(
                "Light / Dark switch Streamlit's native theme (every widget, "
                "table and chart re-renders perfectly). Auto follows your OS setting."
            ),
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
    return remove_pii, max_features, top_gaps, min_score
