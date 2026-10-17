"""Output components for Smart Resume Ranker."""
from __future__ import annotations

import io
from typing import List, Dict

import pandas as pd
import streamlit as st

from utils.matcher import RankedCandidate


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


def render_results(candidates: List[RankedCandidate], meta: Dict[str, float], min_score: int = 0) -> None:
    """Render the results section with rankings, charts, and skill gaps."""
    visible = [c for c in candidates if c.score >= min_score]
    if not visible:
        st.warning("No candidates meet the minimum score filter.")
        st.stop()

    st.markdown('<div class="section-label">Step 02 — Results</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Ranked shortlist</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Scores are TF-IDF cosine similarity (0–100%). '
        "Higher means stronger keyword alignment with the JD.</div>",
        unsafe_allow_html=True,
    )

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
