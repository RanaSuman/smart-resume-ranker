"""Text cleaning, PII redaction, and spaCy lemmatization.

Pipeline: raw text -> PII redaction (optional) -> whitespace / noise
normalization -> spaCy lemmatization + stopword removal.

Batch path uses ``nlp.pipe`` for ~10-50x throughput vs per-doc calls,
keeping preprocessing well under the 500 ms budget for 50 resumes.

Robustness: all entry points tolerate ``None`` / empty input and return
``""``; spaCy failures are logged and degrade gracefully (normalized
fallback text) instead of crashing the batch.
"""

from __future__ import annotations

import logging
import re
from typing import List

logger = logging.getLogger(__name__)

# --- PII / noise patterns (redaction keeps layout, drops identifiers) ---
_EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
_PHONE_RE = re.compile(
    r"(?:\+?\d{1,3}[\s.\-]?)?(?:\(?\d{3,5}\)?[\s.\-]?)?\d{3,5}[\s.\-]?\d{4,6}"
)
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_LINKEDIN_RE = re.compile(r"linkedin\.com/\S+", re.IGNORECASE)
_NON_ALPHA_RE = re.compile(r"[^a-zA-Z\s]")
_MULTI_SPACE_RE = re.compile(r"\s+")


def redact_pii(text: str) -> str:
    """Replace emails, phones, URLs with blanks (no PII retained).

    Args:
        text: Raw input text (``None``-safe).

    Returns:
        Text with PII substrings removed.
    """
    if not text:
        return ""
    try:
        text = _EMAIL_RE.sub(" ", text)
        text = _LINKEDIN_RE.sub(" ", text)
        text = _URL_RE.sub(" ", text)
        text = _PHONE_RE.sub(" ", text)
        return text
    except Exception as exc:  # regex should never fail; be defensive
        logger.warning("PII redaction failed: %s", exc)
        return text or ""


def normalize_whitespace(text: str) -> str:
    """Lowercase, strip non-alpha chars, collapse whitespace.

    Args:
        text: Input text (``None``-safe).

    Returns:
        Normalized lowercase alpha-only string.
    """
    if not text:
        return ""
    try:
        text = text.lower()
        text = _NON_ALPHA_RE.sub(" ", text)
        text = _MULTI_SPACE_RE.sub(" ", text).strip()
        return text
    except Exception as exc:
        logger.warning("Whitespace normalization failed: %s", exc)
        return ""


def get_nlp() -> object:
    """Load spaCy model once (cached via st.cache_resource in app). Parser/NER disabled for speed.

    Raises:
        RuntimeError: If ``en_core_web_sm`` is missing, with install hint,
            or if spaCy itself is not installed.
    """
    try:
        import spacy
    except ImportError as exc:
        raise RuntimeError(
            "spaCy is not installed. Run: pip install -r requirements.txt"
        ) from exc
    try:
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        # Ensure sentencizer-free pipeline stays lean; keep tagger+lemmatizer
        return nlp
    except OSError as exc:
        raise RuntimeError(
            "spaCy model 'en_core_web_sm' not found. "
            "Run: python -m spacy download en_core_web_sm"
        ) from exc


def _lemmatize_with_nlp(nlp: object, text: str) -> str:  # type: ignore[valid-type]
    """Lemmatize a single normalized string (graceful on model errors)."""
    try:
        doc = nlp(text)  # type: ignore[operator]
    except Exception as exc:
        logger.warning("spaCy call failed, returning normalized fallback: %s", exc)
        return text  # graceful degradation: TF-IDF still works
    try:
        tokens = [
            token.lemma_.lower().strip()
            for token in doc  # type: ignore[attr-defined]
            if token.is_alpha and not token.is_stop and len(token.lemma_) > 2
        ]
        return " ".join(tokens)
    except Exception as exc:
        logger.warning("Lemmatization failed: %s", exc)
        return text


def preprocess_text(text: str, remove_pii: bool = True) -> str:
    """Full pipeline for a single document.

    Args:
        text: Raw extracted text.
        remove_pii: Whether to redact emails/phones/URLs first.

    Returns:
        Space-separated lemmatized token string (ready for TF-IDF).
        Empty string for empty / unprocessable input (never raises).
    """
    if not text or not isinstance(text, str) or not text.strip():
        return ""
    try:
        if remove_pii:
            text = redact_pii(text)
        text = normalize_whitespace(text)
        if not text:
            return ""
        nlp = get_nlp()
        return _lemmatize_with_nlp(nlp, text)
    except RuntimeError:
        raise  # missing model / spacy: caller surfaces install hint
    except Exception as exc:
        logger.warning("preprocess_text failed, returning normalized text: %s", exc)
        return normalize_whitespace(text)


def preprocess_corpus(texts: List[str], remove_pii: bool = True) -> List[str]:
    """Efficient batch preprocessing with ``nlp.pipe``.

    Time  : O(total tokens) — single pass, no per-doc model reload.
    Space : O(output tokens) — generators consumed incrementally.

    Args:
        texts: List of raw documents (JD + resumes).
        remove_pii: Redact PII before lemmatization.

    Returns:
        List of lemmatized strings, same order as input. Failures degrade
        to normalized (non-lemmatized) text, never raise except for a
        missing spaCy model (:class:`RuntimeError`).
    """
    if not texts:
        return []
    # Stage 1: cheap regex normalization (no model needed)
    normalized: List[str] = []
    for t in texts:
        try:
            if not t or not isinstance(t, str) or not t.strip():
                normalized.append("")
                continue
            if remove_pii:
                t = redact_pii(t)
            normalized.append(normalize_whitespace(t))
        except Exception as exc:
            logger.warning("Normalization failed for one doc: %s", exc)
            normalized.append("")

    # Stage 2: lemmatization via pipe (batch_size tuned for 50+ docs)
    nlp = get_nlp()  # may raise RuntimeError with install hint — intended
    results: List[str] = [""] * len(normalized)
    non_empty_idx = [i for i, t in enumerate(normalized) if t]
    non_empty = [normalized[i] for i in non_empty_idx]
    if non_empty:
        try:
            for i, doc in zip(
                non_empty_idx, nlp.pipe(non_empty, batch_size=32)  # type: ignore[attr-defined]
            ):
                try:
                    tokens = [
                        token.lemma_.lower().strip()
                        for token in doc
                        if token.is_alpha and not token.is_stop
                        and len(token.lemma_) > 2
                    ]
                    results[i] = " ".join(tokens) or normalized[i]
                except Exception as exc:
                    logger.warning("Lemmatization failed for doc %d: %s", i, exc)
                    results[i] = normalized[i]
        except Exception as exc:
            logger.warning("nlp.pipe batch failed, using normalized fallback: %s", exc)
            for i in non_empty_idx:
                results[i] = normalized[i]
    return results
