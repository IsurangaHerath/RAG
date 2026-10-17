"""PDF loader built on pypdf (Stage 12).

This is the second half of the loader pair started in Stage 11. Where the
text loader deals with *encoding* (UTF-8, BOMs, line endings), a PDF deals
with *structure*: a container of pages, each of which may or may not have a
text layer, and which may be encrypted.

Two design choices are worth stating up front, because they shape everything
else in this module:

**Pages are kept, not flattened.** A PDF is a sequence of pages, and a PDF
is the one format in this project where a citation can plausibly say "see
page 7". Joining every page into one string here and throwing the boundaries
away would make that impossible to recover later, so ``PdfDocument.pages``
is the authoritative structure and ``.text`` is only a convenience join.

**A missing text layer is not an error.** A scanned document is a perfectly
valid PDF that simply contains no text. Slot 14's ingest service is what
decides such a file should be *skipped*, so this loader reports the fact
(``has_text_layer`` is ``False``) rather than raising. Genuine failures —
the bytes are corrupt, or the file is encrypted and we have no password —
still raise a typed error, because those are things the caller may need to
act on.

OCR remains out of scope, and deliberately so: inventing placeholder text
for a scanned page would put fabricated content into the vector index, which
is precisely the failure mode this project exists to avoid.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from stat import S_ISDIR
from typing import Final

from pypdf import PdfReader
from pypdf.errors import (
    EmptyFileError,
    FileNotDecryptedError,
    PdfReadError,
    PdfStreamError,
    PyPdfError,
)

from app.domain.errors import DocumentLoadError, NotFoundError, ValidationError

#: The only media type this module handles.
PDF_MEDIA_TYPE: Final[str] = "application/pdf"


@dataclass(frozen=True, slots=True)
class PdfPage:
    """One page of text, tagged with the number a human would cite.

    ``page_number`` is **1-based**, matching how people count and how
    citations read ("page 3"), even though Python indexes from zero. The
    mismatch is a classic off-by-one bug source, so it is made explicit here
    and covered by a test.
    """

    #: 1-based page number as it appears in a citation.
    page_number: int
    #: Extracted text, or ``""`` when the page has no text layer.
    text: str

    @property
    def char_count(self) -> int:
        """Number of characters of text on this page."""
        return len(self.text)

    @property
    def is_empty(self) -> bool:
        """True when this page yielded no usable text (e.g. a scan)."""
        return not self.text.strip()


@dataclass(frozen=True, slots=True)
class PdfDocument:
    """Every page of a PDF plus how the file was read.

    ``pages`` always has exactly ``page_count`` entries in order, so
    ``pages[i].page_number == i + 1`` holds for every document — including
    ones where individual pages failed to extract, which are listed in
    ``failed_pages`` with empty text. That positional guarantee is what lets
    Stage 18 attach a page number to every chunk without bookkeeping.
    """

    #: All pages in order; length equals :attr:`page_count`.
    pages: tuple[PdfPage, ...]
    #: Always ``application/pdf`` (kept for symmetry with ``TextDocument``).
    media_type: str
    #: Size of the file as read from disk.
    byte_size: int
    #: Whether the source file was encrypted (even if we decrypted it).
    encrypted: bool
    #: 1-based numbers of pages whose text extraction raised.
    failed_pages: tuple[int, ...]
    #: Basename only — never an absolute path.
    source_name: str

    @property
    def page_count(self) -> int:
        """Number of pages in the PDF."""
        return len(self.pages)

    @property
    def text(self) -> str:
        """All pages joined by a blank line.

        A convenience view only; use :attr:`pages` when the page a passage
        came from matters.
        """
        return "\n\n".join(page.text for page in self.pages)

    @property
    def char_count(self) -> int:
        """Number of characters of text across all pages."""
        return sum(page.char_count for page in self.pages)

    @property
    def has_text_layer(self) -> bool:
        """Whether at least one page yielded non-whitespace text."""
        return any(not page.is_empty for page in self.pages)

    @property
    def is_empty(self) -> bool:
        """True when no page yielded usable text.

        This is the scanned-document case. It is not an error: Stage 14
        records such a file as *skipped* with an honest reason.
        """
        return not self.has_text_layer

    def summary(self) -> dict[str, str | int | bool | list[int]]:
        """JSON-safe summary for logs — contains no document text."""
        return {
            "source": self.source_name,
            "media_type": self.media_type,
            "page_count": self.page_count,
            "encrypted": self.encrypted,
            "failed_pages": list(self.failed_pages),
            "char_count": self.char_count,
            "byte_size": self.byte_size,
            "has_text_layer": self.has_text_layer,
            "is_empty": self.is_empty,
        }


@contextmanager
def _quiet_pypdf() -> Iterator[None]:
    """Silence pypdf's warnings for the duration of a read.

    pypdf logs things like ``invalid pdf header`` and ``EOF marker not
    found`` to its own logger before raising. Every one of those turns into
    a typed error carrying the real reason, so leaving them on the root
    logger only produces noise that looks like a crash. The previous level
    is restored afterwards so we do not change global logging state.
    """
    logger = logging.getLogger("pypdf")
    previous = logger.level
    logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        logger.setLevel(previous)


def _check_path(path: Path) -> int:
    """Validate the path and return its size in bytes.

    A single ``stat`` call does the work: it tells us whether the file exists
    (``FileNotFoundError``), whether it is a directory (``S_ISDIR``), and how
    big it is. The text loader reaches the same three answers through
    ``exists``/``is_dir``/``stat``, so this avoids two extra syscalls while
    failing the same conditions with the same error types.
    """
    try:
        info = path.stat()
    except FileNotFoundError as exc:
        raise NotFoundError(
            f"file does not exist: {path.name}",
            code="file_not_found",
            detail=str(path.resolve()),
        ) from exc
    except OSError as exc:
        raise DocumentLoadError(
            f"could not read {path.name}",
            code="unreadable_file",
            detail=str(exc),
        ) from exc

    if S_ISDIR(info.st_mode):
        raise ValidationError(
            f"expected a file but got a directory: {path.name}",
            code="path_is_a_directory",
        )

    return info.st_size


def _translate_pypdf_error(exc: PyPdfError, source_name: str) -> DocumentLoadError:
    """Map a pypdf exception onto our typed hierarchy.

    Order matters: :class:`EmptyFileError` and :class:`FileNotDecryptedError`
    both subclass :class:`PdfReadError`, so the specific ones are checked
    first.
    """
    if isinstance(exc, EmptyFileError):
        return DocumentLoadError(
            f"{source_name} is empty, so it is not a usable PDF.",
            code="invalid_pdf",
            detail=f"{source_name}: empty file",
        )
    if isinstance(exc, FileNotDecryptedError):
        return DocumentLoadError(
            f"{source_name} is encrypted and no password was accepted.",
            code="pdf_encrypted",
            detail=f"{source_name}: FileNotDecryptedError",
        )
    if isinstance(exc, (PdfStreamError, PdfReadError)):
        return DocumentLoadError(
            f"{source_name} could not be parsed as a PDF. The file may be "
            f"corrupt, truncated, or not really a PDF.",
            code="invalid_pdf",
            detail=f"{source_name}: {type(exc).__name__}: {exc}",
        )
    return DocumentLoadError(
        f"{source_name} could not be read as a PDF.",
        code="invalid_pdf",
        detail=f"{source_name}: {type(exc).__name__}: {exc}",
    )


def _open_reader(path: Path, source_name: str) -> PdfReader:
    """Open the PDF, translating pypdf and OS failures into typed errors."""
    try:
        return PdfReader(str(path))
    except PyPdfError as exc:
        raise _translate_pypdf_error(exc, source_name) from exc
    except OSError as exc:
        raise DocumentLoadError(
            f"could not read {source_name}",
            code="unreadable_file",
            detail=str(exc),
        ) from exc


def _decrypt(reader: PdfReader, source_name: str, password: str | None) -> None:
    """Decrypt in place when needed, or raise a typed error.

    pypdf reports a wrong password by *returning* a falsy value rather than
    raising, so the return value has to be checked explicitly — a bare
    ``reader.decrypt(password)`` would silently continue and then fail later
    with a confusing error.
    """
    if not reader.is_encrypted:
        return

    if password is None:
        raise DocumentLoadError(
            f"{source_name} is encrypted. No password was supplied, so its "
            f"contents cannot be read.",
            code="pdf_encrypted",
            detail=f"{source_name}: encrypted, no password given",
        )

    if not reader.decrypt(password):
        raise DocumentLoadError(
            f"{source_name} is encrypted and the supplied password was not accepted.",
            code="pdf_wrong_password",
            detail=f"{source_name}: decryption rejected the provided password",
        )


def _normalize(text: str) -> str:
    """Tidy one page of extracted text.

    Unlike the text loader, which preserves bytes exactly, PDF text is
    already a *reconstruction* of the page — pypdf infers line breaks from
    glyph positions. So normalising line endings and dropping the trailing
    newline that pages usually end with is appropriate here rather than
    destructive.
    """
    return text.replace("\r\n", "\n").replace("\r", "\n").strip("\n")


def _extract_pages(
    reader: PdfReader, source_name: str
) -> tuple[tuple[PdfPage, ...], tuple[int, ...]]:
    """Extract every page, collecting (not hiding) per-page failures.

    A single malformed page should not discard an otherwise readable
    document, but neither should it vanish silently: failed pages are
    recorded in the returned tuple and surface in ``failed_pages``.
    """
    try:
        page_count = len(reader.pages)
    except PyPdfError as exc:
        raise _translate_pypdf_error(exc, source_name) from exc

    pages: list[PdfPage] = []
    failed: list[int] = []

    for index in range(page_count):
        page_number = index + 1
        try:
            raw = reader.pages[index].extract_text() or ""
        except Exception:
            failed.append(page_number)
            pages.append(PdfPage(page_number=page_number, text=""))
        else:
            pages.append(PdfPage(page_number=page_number, text=_normalize(raw)))

    if page_count > 0 and len(failed) == page_count:
        # Nothing at all could be extracted: that is a real failure, not a
        # scanned document, so say so rather than returning an empty result.
        raise DocumentLoadError(
            f"no page of {source_name} could be extracted.",
            code="invalid_pdf",
            detail=f"{source_name}: all {page_count} pages failed extraction",
        )

    return tuple(pages), tuple(failed)


def load_pdf(
    path: str | Path,
    *,
    password: str | None = None,
    media_type: str | None = None,
) -> PdfDocument:
    """Load a PDF into page-tagged text.

    Parameters
    ----------
    path:
        File to read. Must already exist.
    password:
        Password for an encrypted PDF. When ``None`` and the file *is*
        encrypted, a :class:`DocumentLoadError` is raised rather than
        guessing.
    media_type:
        Optional guard. PDFs are the only supported type here, so anything
        else raises :class:`ValidationError`; the parameter exists so the
        eventual multi-format dispatcher can pass a type through.

    Returns
    -------
    PdfDocument
        All pages in order plus provenance (page count, encryption flag,
        byte size, any pages that failed to extract).

    Raises
    ------
    NotFoundError
        The path does not exist.
    ValidationError
        The path is a directory, or a non-PDF media type was requested.
    DocumentLoadError
        The file is empty, corrupt, not really a PDF, encrypted without a
        usable password, or had no extractable pages at all.

    Notes
    -----
    A PDF whose pages exist but contain no text layer (typically a scan) is
    **not** an error: it is returned with ``has_text_layer == False`` so the
    ingest service can skip it honestly. OCR is not attempted.
    """
    file_path = Path(path)
    source_name = file_path.name

    if media_type is not None and media_type != PDF_MEDIA_TYPE:
        raise ValidationError(
            f"load_pdf cannot read media type {media_type!r}; it only handles {PDF_MEDIA_TYPE!r}",
            code="unsupported_media_type",
            detail=f"media_type={media_type}",
        )

    byte_size = _check_path(file_path)

    with _quiet_pypdf():
        reader = _open_reader(file_path, source_name)
        encrypted = bool(reader.is_encrypted)
        _decrypt(reader, source_name, password)
        pages, failed_pages = _extract_pages(reader, source_name)

    return PdfDocument(
        pages=pages,
        media_type=PDF_MEDIA_TYPE,
        byte_size=byte_size,
        encrypted=encrypted,
        failed_pages=failed_pages,
        source_name=source_name,
    )


def can_load_pdf(media_type: str) -> bool:
    """Whether this module can load the given media type."""
    return media_type == PDF_MEDIA_TYPE


__all__ = [
    "PDF_MEDIA_TYPE",
    "PdfDocument",
    "PdfPage",
    "can_load_pdf",
    "load_pdf",
]
