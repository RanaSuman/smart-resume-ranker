"""Secure in-memory PDF / DOCX text extraction.

Security guarantees:
  * Zero disk writes — everything stays in ``io.BytesIO``.
  * Filename sanitization to block path traversal.
  * Strict extension whitelist (``.pdf``, ``.docx``) **plus**
    magic-byte / file-signature validation (MIME sniffing, not just extension).
  * File-size cap to prevent memory exhaustion (DoS).
  * No execution of embedded macros / objects — text only.
  * Extracted-text sanitization (control-char / null-byte stripping) to
    mitigate injection risks when strings are later rendered.

Robustness:
  * Corrupt / encrypted files -> :class:`TextExtractionError` with a clear,
    user-facing message (caller shows ``st.error``); never crashes the app.
  * Scanned / image-only PDFs (``len(text.strip()) < 10``) -> raises
    :class:`ScannedDocumentError` (subclass of :class:`TextExtractionError`)
    so the UI can show a targeted ``st.warning``.
  * PDF engine: ``pdfplumber`` preferred (better layout), ``PyPDF2`` fallback.
    DOCX engine: ``python-docx``; ``BadZipFile`` mapped to a clean error.

Time  : O(P) where P = number of pages / paragraphs.
Space : O(D) where D = decoded text length (streamed, no temp files).
"""

from __future__ import annotations

import io
import logging
import os
import re
import zipfile
from typing import BinaryIO, Tuple

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".docx"})

ALLOWED_MIME_TYPES: frozenset[str] = frozenset(
    {
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }
)

#: Magic-byte signatures: PDF must start with ``%PDF``; DOCX is a ZIP
#: archive and must start with ``PK\x03\x04`` (plus contain
#: ``word/document.xml`` — checked lazily by python-docx on open).
_PDF_MAGIC = b"%PDF"
_DOCX_MAGIC = b"PK\x03\x04"

# 10 MB per file — prevents memory-bloat on batch uploads of 50+ resumes.
MAX_FILE_SIZE_BYTES: int = 10 * 1024 * 1024

#: Threshold below which extracted text is treated as scanned/image-only.
SCANNED_TEXT_THRESHOLD: int = 10

_FILENAME_SAFE_RE = re.compile(r"[^A-Za-z0-9_.\-]")
# Strip C0 controls / DEL (keep \n, \t); kills null bytes & ANSI trickery.
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class FileValidationError(ValueError):
    """Raised when an upload fails security validation."""


class TextExtractionError(RuntimeError):
    """Raised when text extraction fails for a validated file."""


class ScannedDocumentError(TextExtractionError):
    """Raised when a document yields ~no selectable text (scanned image)."""


class EncryptedDocumentError(TextExtractionError):
    """Raised when a document is password-protected / encrypted."""


# ---------------------------------------------------------------------------
# Sanitization / validation helpers
# ---------------------------------------------------------------------------

def sanitize_filename(filename: str, max_len: int = 100) -> str:
    """Strip path components and unsafe chars to prevent path traversal.

    Args:
        filename: Raw client-supplied filename.
        max_len: Maximum length of the returned name.

    Returns:
        A safe basename containing only ``[A-Za-z0-9_.-]``.
    """
    if not filename or not isinstance(filename, str):
        return "unnamed"
    # Drop any directory components: "../../etc/passwd" -> "passwd"
    base = os.path.basename(filename.replace("\\", "/"))
    # Remove null bytes and surrounding whitespace/dots
    base = base.replace("\x00", "").strip().strip(".")
    safe = _FILENAME_SAFE_RE.sub("_", base)
    # Collapse repeated underscores/dots
    safe = re.sub(r"_+", "_", safe)
    safe = safe[:max_len] or "unnamed"
    return safe


def sanitize_text(text: str, max_chars: int = 1_000_000) -> str:
    """Neutralize control characters / null bytes in extracted text.

    Args:
        text: Raw extracted text.
        max_chars: Hard cap to bound downstream memory.

    Returns:
        Sanitized text safe for display / NLP.
    """
    if not text:
        return ""
    text = text.replace("\x00", "")
    text = _CONTROL_CHAR_RE.sub(" ", text)
    if len(text) > max_chars:
        logger.warning(
            "Extracted text truncated: %d chars > cap %d", len(text), max_chars
        )
        text = text[:max_chars]
    return text


def validate_extension(filename: str) -> str:
    """Return lowercase extension if whitelisted, else raise."""
    _, ext = os.path.splitext(filename or "")
    ext = ext.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise FileValidationError(
            f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}"
        )
    return ext


def validate_file_signature(data: bytes, ext: str, filename: str = "") -> str:
    """Validate magic bytes match the claimed extension (MIME sniffing).

    Args:
        data: Raw file bytes.
        ext: Lowercase extension from :func:`validate_extension`.
        filename: For error messages.

    Returns:
        The detected MIME type string.

    Raises:
        FileValidationError: If signature does not match the extension
            (e.g. an ``.exe`` renamed to ``.pdf``).
    """
    if ext == ".pdf":
        if not data.startswith(_PDF_MAGIC):
            raise FileValidationError(
                f"File '{filename}' claims to be PDF but lacks the "
                f"%PDF signature — rejected (possible spoofed type)."
            )
        return "application/pdf"
    if ext == ".docx":
        if not data[:4] == _DOCX_MAGIC:
            raise FileValidationError(
                f"File '{filename}' claims to be DOCX but lacks the "
                f"ZIP (PK..) signature — rejected (possible spoofed type)."
            )
        return (
            "application/vnd.openxmlformats-officedocument."
            "wordprocessingml.document"
        )
    raise FileValidationError(f"Unsupported extension '{ext}'.")  # pragma: no cover


def check_size(data: bytes, filename: str = "") -> None:
    """Enforce per-file size cap."""
    if len(data) > MAX_FILE_SIZE_BYTES:
        raise FileValidationError(
            f"File '{filename}' is {len(data) / 1_048_576:.1f} MB; "
            f"limit is {MAX_FILE_SIZE_BYTES // 1_048_576} MB."
        )
    if len(data) == 0:
        raise FileValidationError(f"File '{filename}' is empty.")


def is_scanned_text(text: str) -> bool:
    """Return True when extracted text looks like a scanned/image-only doc."""
    return len((text or "").strip()) < SCANNED_TEXT_THRESHOLD


# ---------------------------------------------------------------------------
# Engines
# ---------------------------------------------------------------------------

def _extract_pdf_pdfplumber(data: bytes) -> str:
    """Try pdfplumber first (better layout fidelity)."""
    import pdfplumber  # lazy import: optional dependency

    parts: list[str] = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        if not pdf.pages:
            raise TextExtractionError("PDF has no readable pages.")
        for i, page in enumerate(pdf.pages):
            try:
                parts.append(page.extract_text() or "")
            except Exception as exc:  # per-page isolation
                logger.warning("pdfplumber failed on page %d: %s", i, exc)
    return "\n".join(parts).strip()


def _extract_pdf_pypdf2(data: bytes) -> str:
    """Fallback PDF engine using PyPDF2 (always available)."""
    from PyPDF2 import PdfReader
    from PyPDF2.errors import PdfReadError

    try:
        reader = PdfReader(io.BytesIO(data))
    except PdfReadError as exc:
        raise TextExtractionError(f"PDF is corrupted or unreadable: {exc}") from exc

    try:
        if getattr(reader, "is_encrypted", False):
            try:
                result = reader.decrypt("")
                if result == 0:
                    raise EncryptedDocumentError(
                        "PDF is password-protected / encrypted. "
                        "Please upload an unprotected copy."
                    )
            except EncryptedDocumentError:
                raise
            except Exception as exc:
                raise EncryptedDocumentError(
                    "PDF is encrypted and cannot be opened."
                ) from exc
        parts: list[str] = []
        for i, page in enumerate(reader.pages):
            try:
                parts.append(page.extract_text() or "")
            except Exception as exc:
                logger.warning("PyPDF2 failed on page %d: %s", i, exc)
        return "\n".join(parts).strip()
    except (EncryptedDocumentError, TextExtractionError):
        raise
    except Exception as exc:
        raise TextExtractionError(f"PDF parsing failed: {exc}") from exc


def _extract_pdf(data: bytes) -> str:
    """Extract text from PDF bytes with engine fallback + scanned detection."""
    last_error: Exception | None = None
    text = ""
    # 1) Preferred engine.
    try:
        text = _extract_pdf_pdfplumber(data)
    except ImportError:
        logger.debug("pdfplumber not installed; using PyPDF2.")
    except (EncryptedDocumentError, ScannedDocumentError):
        raise
    except TextExtractionError as exc:
        last_error = exc
        logger.warning("pdfplumber extraction failed, trying PyPDF2: %s", exc)
    except Exception as exc:  # corrupt file, bad zip, etc.
        last_error = exc
        logger.warning("pdfplumber unexpected failure, trying PyPDF2: %s", exc)

    # 2) Fallback engine when preferred yielded nothing usable.
    if not text.strip():
        try:
            text = _extract_pdf_pypdf2(data)
        except (EncryptedDocumentError, ScannedDocumentError):
            raise
        except TextExtractionError as exc:
            # Prefer the most informative error.
            raise exc from last_error

    text = sanitize_text(text)
    if is_scanned_text(text):
        raise ScannedDocumentError(
            "This PDF appears to contain scanned images rather than selectable "
            "text (extracted < 10 characters). Please upload a text-based PDF, "
            "a DOCX export, or run OCR first."
        )
    if not text:
        raise TextExtractionError("No extractable text found in PDF.")
    return text


def _extract_docx(data: bytes) -> str:
    """Extract text from DOCX bytes using python-docx (in-memory)."""
    try:
        import docx
    except ImportError as exc:  # pragma: no cover
        raise TextExtractionError(
            "python-docx is not installed. Run: pip install -r requirements.txt"
        ) from exc

    try:
        doc = docx.Document(io.BytesIO(data))
        parts = [p.text for p in doc.paragraphs]
        # Also capture tables (skills matrices often live there)
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    parts.append(cell.text)
        text = sanitize_text("\n".join(t for t in parts if t).strip())
        if is_scanned_text(text):
            raise ScannedDocumentError(
                "This DOCX appears to contain no readable text "
                "(extracted < 10 characters). It may be image-only."
            )
        if not text:
            raise TextExtractionError("No extractable text found in DOCX.")
        return text
    except ScannedDocumentError:
        raise
    except TextExtractionError:
        raise
    except zipfile.BadZipFile as exc:
        logger.warning("DOCX BadZipFile (corrupt/not a real docx): %s", exc)
        raise TextExtractionError(
            "DOCX is corrupted or not a valid Word document "
            "(BadZipFile). Please re-save and re-upload."
        ) from exc
    except Exception as exc:
        logger.warning("DOCX parsing failed: %s", exc)
        raise TextExtractionError(f"DOCX parsing failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_text(data: bytes, filename: str) -> str:
    """Validate and extract text from raw bytes — never touches disk.

    Performs, in order: size check -> extension whitelist ->
    magic-byte signature check -> engine extraction -> text sanitization.

    Args:
        data: Raw file bytes (already read into memory).
        filename: Original client filename (used only for type dispatch).

    Returns:
        Extracted plain text.

    Raises:
        FileValidationError: Spoofed type / bad extension / oversize.
        EncryptedDocumentError: Password-protected file.
        ScannedDocumentError: Image-only document (~no selectable text).
        TextExtractionError: Corrupt file or empty extraction.
    """
    safe_name = sanitize_filename(filename)
    check_size(data, safe_name)
    ext = validate_extension(safe_name)
    try:
        validate_file_signature(data, ext, safe_name)
    except FileValidationError:
        logger.warning("Signature mismatch rejected: %s", safe_name)
        raise
    try:
        if ext == ".pdf":
            return _extract_pdf(data)
        return _extract_docx(data)
    except (ScannedDocumentError, EncryptedDocumentError,
            TextExtractionError, FileValidationError):
        raise
    except Exception as exc:  # defensive: never leak a raw traceback to UI
        logger.exception("Unexpected extraction failure for %s", safe_name)
        raise TextExtractionError(f"Failed to extract '{safe_name}': {exc}") from exc


def extract_text_from_fileobj(fileobj: BinaryIO, filename: str) -> Tuple[str, str]:
    """Convenience wrapper for file-like objects (e.g. Streamlit uploads).

    Reads the stream in chunks (bounded memory), then extracts text.

    Returns:
        (safe_filename, extracted_text)
    """
    safe_name = sanitize_filename(filename)
    validate_extension(safe_name)

    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = fileobj.read(64 * 1024)  # 64 KiB streaming chunks
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_FILE_SIZE_BYTES:
            raise FileValidationError(
                f"File '{safe_name}' exceeds "
                f"{MAX_FILE_SIZE_BYTES // 1_048_576} MB limit."
            )
        chunks.append(chunk)
    data = b"".join(chunks)
    # Release chunk list promptly to bound peak memory
    del chunks
    if not data:
        raise FileValidationError(f"File '{safe_name}' is empty.")
    return safe_name, extract_text(data, safe_name)
