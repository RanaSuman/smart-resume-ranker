"""TF-IDF vectorization + cosine-similarity ranking engine.

Design notes:
  * All matrices stay ``scipy.sparse.csr_matrix`` — never densified,
    so 50+ resumes use O(nnz) memory, not O(docs x vocab).
  * Overall complexity O(N * M) where N = vocab size, M = avg doc length,
    dominated by sparse TF-IDF + sparse cosine similarity.
  * Skill gaps = top TF-IDF terms in JD absent from the resume token set.
  * Zero-vocabulary overlap / empty-vocabulary edge cases return 0.0
    scores instead of raising (division-by-zero / empty-matrix safe).

Time  : O(N * M) sparse vectorize + score (< 500 ms for 50 resumes).
Space : O(nnz) sparse — ``toarray()``/``todense()`` never called on TF-IDF.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class RankedCandidate:
    """Single ranking result."""

    name: str
    score: float  # 0-100 match percentage
    raw_cosine: float  # 0-1 cosine similarity
    rank: int = 0
    skill_gaps: List[str] = field(default_factory=list)


def build_vectorizer(max_features: int = 10_000) -> TfidfVectorizer:
    """Create a tuned TF-IDF vectorizer (inputs are pre-lemmatized).

    Args:
        max_features: Vocabulary cap for speed / memory control.

    Returns:
        Configured :class:`TfidfVectorizer`.
    """
    return TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),  # unigrams + bigrams capture "machine learning"
        max_features=max_features,
        sublinear_tf=True,  # log-scaled tf dampens keyword stuffing
        norm="l2",
        token_pattern=r"(?u)\b[a-z][a-z]+\b",
    )


def _zero_result(
    resume_names: Sequence[str], top_gap_terms: int, reason: str,
    vectorize_ms: float = 0.0,
) -> tuple[List[RankedCandidate], Dict[str, float]]:
    """Build an all-zero fallback ranking (zero-overlap / empty-vocab safe)."""
    logger.warning("rank_resumes fallback (%s): all scores 0.0", reason)
    candidates = [
        RankedCandidate(name=n, score=0.0, raw_cosine=0.0, skill_gaps=[])
        for n in resume_names
    ]
    for rank, c in enumerate(candidates, start=1):
        c.rank = rank
    meta = {
        "vectorize_and_score_ms": round(vectorize_ms, 2),
        "n_resumes": float(len(candidates)),
        "vocab_size": 0.0,
        "fallback": reason,
    }
    return candidates, meta


def rank_resumes(
    jd_processed: str,
    resumes_processed: Sequence[str],
    resume_names: Sequence[str],
    max_features: int = 10_000,
    top_gap_terms: int = 15,
) -> tuple[List[RankedCandidate], Dict[str, float]]:
    """Rank resumes against a JD with cosine similarity.

    Args:
        jd_processed: Preprocessed JD string.
        resumes_processed: Preprocessed resume strings (same order as names).
        resume_names: Display names for candidates.
        max_features: TF-IDF vocabulary cap.
        top_gap_terms: How many top JD terms to consider for skill gaps.

    Returns:
        (ranked_candidates_desc, meta) where meta holds timing info.
        Zero-overlap / empty-vocabulary inputs yield 0.0 scores, not errors.

    Raises:
        ValueError: On empty JD or mismatched inputs (genuinely
            un-runnable; zero overlap alone does NOT raise).
    """
    if not jd_processed or not jd_processed.strip():
        raise ValueError("Job description is empty after preprocessing.")
    if len(resumes_processed) == 0:
        raise ValueError("No resumes provided.")
    if len(resumes_processed) != len(resume_names):
        raise ValueError("resumes and names length mismatch.")

    t0 = time.perf_counter()

    # Guard: resumes that preprocess to empty still need a placeholder token
    # so the vectorizer keeps row alignment; they will score ~0.
    safe_resumes = [
        r if isinstance(r, str) and r.strip() else "empty document placeholder"
        for r in resumes_processed
    ]

    vectorizer = build_vectorizer(max_features=max_features)
    corpus = [jd_processed] + list(safe_resumes)
    try:
        tfidf: csr_matrix = vectorizer.fit_transform(corpus)  # sparse, never dense
    except ValueError as exc:
        # sklearn raises "empty vocabulary" when no token survives
        # (e.g. JD + resumes share zero usable tokens / all stopwords).
        elapsed = (time.perf_counter() - t0) * 1000.0
        return _zero_result(
            resume_names, top_gap_terms, f"empty vocabulary: {exc}", elapsed
        )

    try:
        jd_vec = tfidf[0]
        resume_mat = tfidf[1:]

        # Sparse cosine similarity: returns small (1 x R) dense row — safe.
        # Never call .toarray() on the full TF-IDF matrix.
        # NaN-safe: zero-norm rows yield NaN -> replaced with 0.0.
        with np.errstate(divide="ignore", invalid="ignore"):
            sims: np.ndarray = np.nan_to_num(
                cosine_similarity(jd_vec, resume_mat).ravel(), nan=0.0
            )
    except Exception as exc:
        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.exception("Cosine similarity failed: %s", exc)
        return _zero_result(resume_names, top_gap_terms, f"score error: {exc}", elapsed)

    vectorize_ms = (time.perf_counter() - t0) * 1000.0

    try:
        feature_names = vectorizer.get_feature_names_out()
    except Exception:
        feature_names = np.array([])

    # JD term weights for gap analysis (sparse row -> sorted indices).
    try:
        jd_row = jd_vec.tocoo()
        if jd_row.data.size == 0:
            jd_top_terms: List[str] = []
        else:
            jd_ranked_idx = np.argsort(-jd_row.data)[: top_gap_terms * 3]
            jd_top_terms = [feature_names[jd_row.col[i]] for i in jd_ranked_idx]
    except Exception as exc:
        logger.warning("Skill-gap extraction failed: %s", exc)
        jd_top_terms = []

    candidates: List[RankedCandidate] = []
    for i, (name, sim) in enumerate(zip(resume_names, sims)):
        try:
            raw = float(np.clip(sim, 0.0, 1.0))
        except Exception:
            raw = 0.0
        try:
            resume_text = safe_resumes[i]
            # Build a token set that covers both unigrams AND bigrams so that
            # gap terms from the TF-IDF vocabulary (which may be bigrams like
            # "machine learning") are correctly found in the resume text.
            resume_unigrams = set(resume_text.split())
            resume_words = resume_text.split()
            resume_bigrams = {
                f"{resume_words[j]} {resume_words[j + 1]}"
                for j in range(len(resume_words) - 1)
            }
            resume_ngrams = resume_unigrams | resume_bigrams
            gaps = [t for t in jd_top_terms if t not in resume_ngrams][:top_gap_terms]
        except Exception:
            gaps = []
        candidates.append(
            RankedCandidate(
                name=name, score=round(raw * 100.0, 2), raw_cosine=raw, skill_gaps=gaps
            )
        )

    # Rank descending by score; stable for ties
    candidates.sort(key=lambda c: c.score, reverse=True)
    for rank, c in enumerate(candidates, start=1):
        c.rank = rank

    meta = {
        "vectorize_and_score_ms": round(vectorize_ms, 2),
        "n_resumes": float(len(candidates)),
        "vocab_size": float(len(feature_names)),
    }
    if all(c.score == 0.0 for c in candidates):
        meta["fallback"] = "zero overlap: no common tokens between JD and resumes"
        logger.warning("Zero-vocabulary overlap: all scores 0.0")
    return candidates, meta


def get_score_distribution(
    candidates: Sequence[RankedCandidate], bins: int = 10
) -> Dict[str, List[float]]:
    """Bin match scores for dashboard histogram (pure-python, no extra deps)."""
    scores = [c.score for c in candidates]
    if not scores:
        return {"bin_edges": [], "counts": []}
    try:
        hist, edges = np.histogram(scores, bins=bins, range=(0, 100))
        return {
            "bin_edges": [float(e) for e in edges],
            "counts": [float(h) for h in hist],
        }
    except Exception as exc:
        logger.warning("Score distribution failed: %s", exc)
        return {"bin_edges": [], "counts": []}
