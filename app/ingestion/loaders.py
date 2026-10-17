"""Text file loaders for TXT and Markdown (Stage 11).

Responsibilities, deliberately narrow:

* Read bytes from disk and decode them as UTF-8.
* Normalise line endings so Windows, macOS and Linux files all behave the
  same downstream.
* Preserve everything else — indentation, blank lines, headings, code
  fences. Chunking (Stage 18) depends on that structure, and rewriting it
  here would silently damage Markdown.
* Fail with a clear, typed error when a file genuinely cannot be read.

What this module explicitly does **not** do: OCR. A scanned PDF or an image
has no extractable text, and inventing placeholder text for it would poison
the index with fabricated content. Those cases raise a typed error so the
ingestion service can mark the file *skipped* with an honest reason.
"""

from __future__ import annotations

import codecs
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from app.domain.errors import DocumentLoadError, NotFoundError, ValidationError

#: Media types this module can load. PDF is handled separately (Stage 12).
TEXT_MEDIA_TYPES: Final[frozenset[str]] = frozenset({"text/plain", "text/markdown"})

#: Media type used when a caller does not supply one; derived from the suffix.
_DEFAULT_MEDIA_TYPE: Final[str] = "text/plain"

# UTF-8 decoding is strict on purpose: silently replacing undecodable bytes
# with U+FFFD would put corrupt text into the vector index, where it is very
# hard to notice later.


@dataclass(frozen=True, slots=True)
class TextDocument:
    """Decoded text plus the facts about how it was decoded.

    Separated from :class:`app.domain.models.Document` because loading is
    about *bytes on disk*; the domain record is about the logical document
    after ingestion. Keeping them apart lets callers decide what to do with
    an empty file without the loader having to know about ingest status.
    """

    #: Decoded text with normalised (``\\n``) line endings.
    text: str
    #: Effective media type, e.g. ``text/markdown``.
    media_type: str
    #: Encoding actually used, e.g. ``utf-8`` or ``utf-8-sig``.
    encoding: str
    #: Whether a byte-order mark was present and stripped.
    had_bom: bool
    #: Line ending style found in the original file.
    original_newlines: str
    #: Whether line endings were rewritten to ``\\n``.
    newlines_normalized: bool
    #: Size of the file as read from disk.
    byte_size: int
    #: Basename only — never an absolute path.
    source_name: str

    @property
    def char_count(self) -> int:
        """Number of characters of decoded text."""
        return len(self.text)

    @property
    def line_count(self) -> int:
        """Number of lines (1 for a single-line file, 0 for empty text)."""
        if not self.text:
            return 0
        return self.text.count("\n") + 1

    @property
    def is_empty(self) -> bool:
        """True when the file contains no extractable text.

        Empty is *valid*, not an error: Stage 14 reports such files as
        skipped rather than failing the whole batch.
        """
        return not self.text.strip()

    def summary(self) -> dict[str, str | int | bool]:
        """JSON-safe summary for logs — contains no document text."""
        return {
            "source": self.source_name,
            "media_type": self.media_type,
            "encoding": self.encoding,
            "had_bom": self.had_bom,
            "original_newlines": self.original_newlines,
            "newlines_normalized": self.newlines_normalized,
            "byte_size": self.byte_size,
            "char_count": self.char_count,
            "line_count": self.line_count,
            "is_empty": self.is_empty,
        }


def media_type_for_path(path: str | Path) -> str:
    """Map a file suffix to a media type, defaulting to ``text/plain``.

    Discovery already filters to supported extensions, so defaulting here
    keeps the loader usable for ad-hoc files (a test fixture, a temp file)
    without demanding an extension every time.
    """
    suffix = Path(path).suffix.lower()
    if suffix == ".md":
        return "text/markdown"
    if suffix == ".pdf":
        raise ValidationError(
            "PDF files must be loaded with the PDF loader (app.ingestion.loaders.load_pdf)",
            code="unsupported_media_type",
            detail=f"suffix={suffix}",
        )
    return _DEFAULT_MEDIA_TYPE


def detect_newline_style(raw: bytes) -> str:
    """Classify the dominant line ending in ``raw``.

    Returns ``crlf``, ``cr``, ``lf``, ``mixed``, or ``none``.
    """
    crlf = raw.count(b"\r\n")
    # Lone CRs: total CRs minus those that were part of a CRLF pair.
    cr = raw.count(b"\r") - crlf
    lf = raw.count(b"\n") - crlf

    styles = [name for name, count in (("crlf", crlf), ("cr", cr), ("lf", lf)) if count]

    if not styles:
        return "none"
    if len(styles) > 1:
        return "mixed"
    return styles[0]


def _read_bytes(path: Path) -> bytes:
    """Read the file, translating OS errors into our typed hierarchy."""
    if not path.exists():
        raise NotFoundError(
            f"file does not exist: {path.name}",
            code="file_not_found",
            detail=str(path.resolve()),
        )
    if path.is_dir():
        raise ValidationError(
            f"expected a file but got a directory: {path.name}",
            code="path_is_a_directory",
        )

    try:
        return path.read_bytes()
    except PermissionError as exc:
        raise DocumentLoadError(
            f"permission denied reading {path.name}",
            code="permission_denied",
            detail=str(exc),
        ) from exc
    except OSError as exc:
        raise DocumentLoadError(
            f"could not read {path.name}",
            code="unreadable_file",
            detail=str(exc),
        ) from exc


def _decode(raw: bytes, *, source_name: str) -> tuple[str, str, bool]:
    """Decode bytes as UTF-8, returning ``(text, encoding, had_bom)``.

    Raises
    ------
    DocumentLoadError
        If the bytes are not valid UTF-8, or if they contain NUL bytes
        (a strong sign a binary file was given a ``.txt``/``.md`` name).
    """
    had_bom = raw.startswith(codecs.BOM_UTF8)

    # A NUL byte means this is almost certainly binary (e.g. a .docx or an
    # image renamed to .txt). Decoding would "succeed" with garbage, so we
    # reject it explicitly rather than index nonsense.
    if b"\x00" in raw:
        raise DocumentLoadError(
            f"{source_name} contains NUL bytes, so it is binary data rather "
            f"than UTF-8 text. It was probably mislabelled with a text extension.",
            code="binary_content",
            detail=f"{source_name}: NUL byte found",
        )

    encoding = "utf-8-sig" if had_bom else "utf-8"
    try:
        # utf-8-sig strips the BOM on decode; plain utf-8 decodes normally.
        text = raw.decode(encoding)
    except UnicodeDecodeError as exc:
        raise DocumentLoadError(
            f"{source_name} is not valid UTF-8 text (bad byte at offset "
            f"{exc.start}). Re-save it as UTF-8 and try again.",
            code="encoding_error",
            detail=f"{source_name}: {exc.reason} at offset {exc.start}",
        ) from exc

    return text, ("utf-8-sig" if had_bom else "utf-8"), had_bom


def _normalize_newlines(text: str) -> str:
    """Rewrite every line ending to ``\\n``.

    CRLF is handled first so ``\\r\\n`` never becomes ``\\n\\n``.
    """
    return text.replace("\r\n", "\n").replace("\r", "\n")


def load_text(
    path: str | Path,
    *,
    media_type: str | None = None,
    normalize_newlines: bool = True,
) -> TextDocument:
    """Load a ``.txt`` or ``.md`` file.

    Parameters
    ----------
    path:
        File to read. Must already exist.
    media_type:
        Override the inferred type; defaults to the value from the suffix.
    normalize_newlines:
        Convert CRLF/CR to LF (default). Set ``False`` to inspect the file
        exactly as stored — useful in tests and diagnostics.

    Returns
    -------
    TextDocument
        Decoded text and provenance (encoding, BOM, original newlines).

    Raises
    ------
    NotFoundError
        The path does not exist, or is a directory (see below).
    ValidationError
        The path is a directory, or an unsupported media type was requested.
    DocumentLoadError
        Permission denied, an OS read failure, invalid UTF-8, or binary
        content masquerading as text.

    Notes
    -----
    Formatting other than line endings is preserved byte-for-byte. Indented
    code blocks, blank lines and Markdown structure survive intact because
    chunking relies on them.
    """
    file_path = Path(path)
    source_name = file_path.name

    resolved_type = media_type if media_type is not None else media_type_for_path(file_path)
    if resolved_type not in TEXT_MEDIA_TYPES:
        raise ValidationError(
            f"load_text cannot read media type {resolved_type!r}; "
            f"supported types are {sorted(TEXT_MEDIA_TYPES)}",
            code="unsupported_media_type",
            detail=f"media_type={resolved_type}",
        )

    raw = _read_bytes(file_path)
    text, encoding, had_bom = _decode(raw, source_name=source_name)
    newline_style = detect_newline_style(raw)

    if normalize_newlines and newline_style in {"crlf", "cr", "mixed"}:
        text = _normalize_newlines(text)
        normalized = True
    else:
        normalized = False

    return TextDocument(
        text=text,
        media_type=resolved_type,
        encoding=encoding,
        had_bom=had_bom,
        original_newlines=newline_style,
        newlines_normalized=normalized,
        byte_size=len(raw),
        source_name=source_name,
    )


def load(path: str | Path, *, media_type: str | None = None) -> TextDocument:
    """Convenience alias for :func:`load_text`, named to match the Stage 14 API."""
    return load_text(path, media_type=media_type)


def can_load(media_type: str) -> bool:
    """Whether this module has a loader for the given media type."""
    return media_type in TEXT_MEDIA_TYPES


__all__ = [
    "TEXT_MEDIA_TYPES",
    "TextDocument",
    "can_load",
    "detect_newline_style",
    "load",
    "load_text",
    "media_type_for_path",
]
