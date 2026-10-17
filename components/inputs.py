"""Input component for Smart Resume Ranker."""
from __future__ import annotations

import streamlit as st

from config import MAX_RESUMES
from utils.parser import sanitize_filename, extract_text


def render_inputs() -> tuple[str, list]:
    """Render the upload input sections for JD and resumes.

    Returns:
        (jd_text, resume_files) where jd_text is the job description text
        and resume_files is the list of uploaded resume file objects.
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
                f"**2️⃣ &nbsp; Candidate resumes** <span style='color:#64748B'>(up to {MAX_RESUMES})</span>"
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
                    from utils.parser import (
                        ScannedDocumentError,
                        EncryptedDocumentError,
                        FileValidationError,
                        TextExtractionError,
                    )

                    jd_text = extract_text(raw, fname)
                except Exception as exc:
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

    return jd_text, resume_files
