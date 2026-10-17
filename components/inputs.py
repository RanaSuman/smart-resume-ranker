"""Input component for Smart Resume Ranker."""
from __future__ import annotations

import logging

import streamlit as st

from config import MAX_RESUMES
from utils.parser import (
    ALLOWED_EXTENSIONS,
    EncryptedDocumentError,
    FileValidationError,
    ScannedDocumentError,
    TextExtractionError,
    extract_text,
    sanitize_filename,
)

logger = logging.getLogger(__name__)


def render_inputs() -> tuple[str | None, list]:
    """Render the upload input sections for JD and resumes.

    Returns:
        (jd_text, resume_files) where jd_text is the extracted job description
        text (or None if no input provided yet) and resume_files is the list
        of uploaded resume file objects.

    Note:
        This function renders the upload widgets and extracts the JD text, but
        does NOT call st.stop() — validation and stopping is the caller's
        responsibility so that this component can be used purely for rendering.
    """
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
            st.markdown(
                f"**2️⃣ &nbsp; Candidate resumes** <span style='color:#64748B'>(up to {MAX_RESUMES})</span>",
                unsafe_allow_html=True,
            )
            st.caption("PDF or DOCX · each file processed in memory, in isolation.")
            resume_files = st.file_uploader(
                "Upload resumes",
                type=["pdf", "docx"],
                accept_multiple_files=True,
                key="resumes",
            )
            if resume_files:
                if len(resume_files) > MAX_RESUMES:
                    st.warning(f"Only the first {MAX_RESUMES} files will be processed.")
                    resume_files = resume_files[:MAX_RESUMES]
                st.success(f"✅ {len(resume_files)} resume(s) ready for ranking.")
            else:
                st.info("No resumes yet — upload at least one to enable ranking.")

    # Return raw file objects; JD extraction is done in app.py on button click.
    return jd_file, jd_pasted, resume_files
