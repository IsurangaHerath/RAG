"""Ingest service: the point where discovery, loading and metadata meet.

Until now each piece has been built and tested in isolation — discovery
classifies files without opening them, the loaders read one file, metadata
describes one document. Nothing yet says *"here is a directory, tell me
what would be indexed"*. That is this module's whole job: run the three
steps in order for every file found, and report honestly on each one.

::

    discover()  ->  load each file  ->  build_metadata()  ->  FileResult

Three decisions are worth spelling out, because each one could reasonably
have gone the other way.

**Outcomes are not lifecycle statuses.** Every result is tagged
:class:`FileOutcome` — ``loaded``, ``skipped`` or ``failed`` — which is a
deliberately *different* vocabulary from
:class:`app.domain.models.IngestStatus`. That enum describes a registered
document travelling through a pipeline (``pending -> parsing -> indexing ->
indexed``), and it is owned by the registry. This service runs before any
registry exists, and its three outcomes do not map onto those five:

* A *skipped* file never becomes a document at all. There is no record to
  advance, so forcing it into a lifecycle status would invent one.
* A *failed* file failed to be **read**, which is not the same event as a
  document failing to be **indexed**. Conflating them would make a corrupt
  upload look like a vector-store outage.

Keeping the two apart means the later registry code can map these outcomes
onto statuses deliberately, instead of discovering that a skipped file
cannot legally reach any terminal state in the existing state machine.

**Only deliberate errors are caught.** The loaders already translate the
messy real world — missing files, permission errors, corrupt PDFs, rejected
passwords — into :class:`~app.domain.errors.RAGError` with a stable code, so
catching that one type is enough to turn a bad file into a ``failed`` result
and carry on with the rest of the batch. Anything *else* escaping is a
programming bug, and bugs must not be converted into a tidy status line:
that is exactly how a defect survives unnoticed until production. One broken
file therefore never stops the batch, but a broken *code path* still fails
loudly.

**Empty is skipped, not failed.** A text file with no content, or a scanned
PDF with no text layer, is a perfectly valid file — the loaders already
report ``is_empty`` rather than raising, precisely so this service can mark
it ``skipped`` with a reason. Rejecting it as a failure would blame the user
for a file their own tool produced, and would bury real failures in noise.
OCR stays out of scope for the reason it always has: substituting
placeholder text would put fabricated content into the index.

**What is deliberately not here.** No ``doc_id``, no content hash, no
``Document`` record and no registry writes — those are deduplication's job
and come next. Holding them back keeps this module a pure *function of a
file* (path in, outcome out), which is what makes it trivially testable
with no store, no network and no API key.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

from app.domain.errors import RAGError, ValidationError
from app.ingestion.discovery import DiscoveryResult, UnsupportedFile, discover
from app.ingestion.loaders import (
    TEXT_MEDIA_TYPES,
    TextDocument,
    can_load,
    load_text,
    media_type_for_path,
)
from app.ingestion.metadata import DocumentMetadata, build_metadata
from app.ingestion.pdf_loader import (
    PDF_MEDIA_TYPE,
    PdfDocument,
    can_load_pdf,
    load_pdf,
)

#: Every media type this service knows how to read.
LOADABLE_MEDIA_TYPES: Final[frozenset[str]] = frozenset(TEXT_MEDIA_TYPES) | {PDF_MEDIA_TYPE}

#: Stable code for a text file that had nothing in it. Uses the same
#: ``snake_case`` convention as the error codes so reports can group skip
#: reasons and error codes in one column.
EMPTY_FILE_CODE: Final[str] = "empty_file"

#: Stable code for a PDF with no extractable text (a scan). Kept distinct
#: from :data:`EMPTY_FILE_CODE` because the two mean very different things to
#: a user: one is "you uploaded a blank file", the other is "this PDF would
#: need OCR".
NO_TEXT_LAYER_CODE: Final[str] = "no_text_layer"


class FileOutcome(StrEnum):
    """What happened to one file during a pass over it.

    ``StrEnum`` so it serialises straight to JSON and compares equal to its
    string form (``result.outcome == "skipped"`` works).

    Deliberately *not* :class:`app.domain.models.IngestStatus` — see the
    module docstring. These describe a *file*; the lifecycle statuses
    describe a *registered document*.
    """

    #: Read successfully; metadata was built and it has text to index.
    LOADED = "loaded"
    #: A valid file with no extractable text — nothing to index, no error.
    SKIPPED = "skipped"
    #: Could not be read. ``code`` carries the typed error's code.
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class FileResult:
    """The outcome for exactly one file.

    Immutable, like every model in this project, so a report built from a
    batch cannot have its underlying results rewritten underneath it while
    it is still being rendered.

    Parameters
    ----------
    source_name:
        Basename only — never a path, and never an absolute one.
    relative_path:
        Forward-slashed path relative to the scanned root, so it reads the
        same on Windows and POSIX.
    media_type:
        What the file turned out to be, not what its name claimed.
    byte_size:
        Size of the file as read. ``None`` only when the load failed before
        the size was ever learned *and* no caller supplied one.
    outcome:
        ``loaded`` / ``skipped`` / ``failed``.
    code:
        Stable machine-readable reason, present for every non-loaded
        result: a skip code (``empty_file``, ``no_text_layer``) or — for a
        failure — exactly the :attr:`~app.domain.errors.RAGError.code` the
        loader raised, so one column can be grouped across both kinds.
        ``None`` when loaded, because success needs no explanation.
    reason:
        Human-readable explanation for the same non-loaded result, suitable
        for showing to a user. ``None`` when loaded.
    document:
        The loaded document when one was produced. Present for ``loaded``
        *and* ``skipped`` — a skipped file loaded fine, it simply had no
        text — and ``None`` when the load failed.
    metadata:
        Metadata built for the document; same availability as
        ``document``.
    page_count:
        Pages for PDFs, ``None`` for plain text.
    """

    source_name: str
    relative_path: str
    media_type: str
    byte_size: int | None
    outcome: FileOutcome
    code: str | None = None
    reason: str | None = None
    document: TextDocument | PdfDocument | None = None
    metadata: DocumentMetadata | None = None
    page_count: int | None = None

    @property
    def is_loaded(self) -> bool:
        """True when this file has text and metadata ready to be indexed."""
        return self.outcome is FileOutcome.LOADED

    @property
    def is_skipped(self) -> bool:
        """True when this file is valid but has nothing to index."""
        return self.outcome is FileOutcome.SKIPPED

    @property
    def is_failed(self) -> bool:
        """True when this file could not be read."""
        return self.outcome is FileOutcome.FAILED

    @property
    def is_processed(self) -> bool:
        """True when the file was read, whether or not it had text.

        The complement of a *load* failure, so callers can say "4 of 5 files
        were read" without counting skips as failures.
        """
        return self.outcome is not FileOutcome.FAILED

    @property
    def text(self) -> str:
        """Extracted text, or ``""`` when there is no loaded document.

        A convenience for the next steps; ``""`` for a failure is
        deliberate rather than an accident, because an empty string is the
        honest answer to "what text did we get out of this file".
        """
        if self.document is None:
            return ""
        return self.document.text

    def summary(self) -> dict[str, object]:
        """JSON-safe summary for logs, reports and API responses.

        Contains provenance, counts and the outcome — **never** document
        text. The title is included because it is sanitised metadata rather
        than body content (see :mod:`app.ingestion.metadata`).
        """
        payload: dict[str, object] = {
            "source_name": self.source_name,
            "relative_path": self.relative_path,
            "media_type": self.media_type,
            "byte_size": self.byte_size,
            "outcome": str(self.outcome),
        }
        if self.code is not None:
            payload["code"] = self.code
        if self.reason is not None:
            payload["reason"] = self.reason
        if self.page_count is not None:
            payload["page_count"] = self.page_count
        if self.metadata is not None:
            payload["title"] = self.metadata.title
            payload["word_count"] = self.metadata.word_count
        return payload


@dataclass(frozen=True, slots=True)
class IngestResult:
    """Everything one pass over a root produced.

    Carries the unsupported files discovery found alongside the ones that
    were attempted, so a single object can answer "what is in this
    directory and what would happen to each file?" — the unsupported list is
    *not* folded into ``files``, because those files were never opened and
    calling them failures would be dishonest.

    Parameters
    ----------
    root:
        The root's **name only** (``"data"``), never an absolute path. An
        absolute server path must not escape into a log line or a response
        a client can see.
    files:
        One result per supported file, in discovery order.
    unsupported:
        Files with an extension we do not handle, untouched by this service.
    """

    root: str
    files: tuple[FileResult, ...]
    unsupported: tuple[UnsupportedFile, ...]

    @property
    def loaded_count(self) -> int:
        """Files with text ready to index."""
        return sum(1 for f in self.files if f.is_loaded)

    @property
    def skipped_count(self) -> int:
        """Valid files that had no extractable text."""
        return sum(1 for f in self.files if f.is_skipped)

    @property
    def failed_count(self) -> int:
        """Files that could not be read."""
        return sum(1 for f in self.files if f.is_failed)

    @property
    def unsupported_count(self) -> int:
        """Files discovery declined to consider at all."""
        return len(self.unsupported)

    @property
    def processed_count(self) -> int:
        """Files this service actually attempted to read."""
        return len(self.files)

    @property
    def total_count(self) -> int:
        """Every file discovered, attempted or not."""
        return len(self.files) + len(self.unsupported)

    @property
    def is_empty(self) -> bool:
        """True when the root held nothing recognisable as a document.

        Matches :attr:`app.ingestion.discovery.DiscoveryResult.is_empty`, so
        a directory containing only a ``.gitkeep`` reads as empty here too.
        """
        return not self.files and not self.unsupported

    @property
    def has_failures(self) -> bool:
        """True when at least one file could not be read.

        Useful as a process exit signal: a batch that loaded 99 files and
        failed one is not a *successful* run, even though it is not a
        total failure either.
        """
        return self.failed_count > 0

    @property
    def indexable_count(self) -> int:
        """Alias for :attr:`loaded_count` — how many files would be indexed."""
        return self.loaded_count

    def relative_paths(self) -> tuple[str, ...]:
        """Attempted files' relative paths in scan order.

        Stable across runs on the same input, which is what makes batch
        results testable without depending on outcome ordering logic.
        """
        return tuple(f.relative_path for f in self.files)

    def failed(self) -> tuple[FileResult, ...]:
        """Only the files that could not be read, in scan order."""
        return tuple(f for f in self.files if f.is_failed)

    def skipped(self) -> tuple[FileResult, ...]:
        """Only the files with no extractable text, in scan order."""
        return tuple(f for f in self.files if f.is_skipped)

    def summary(self) -> dict[str, object]:
        """JSON-safe summary for logs and API responses.

        Contains counts, the root *name*, and one entry per problem file —
        never absolute paths and never document text.
        """
        return {
            "root": self.root,
            "total": self.total_count,
            "processed": self.processed_count,
            "loaded": self.loaded_count,
            "skipped": self.skipped_count,
            "failed": self.failed_count,
            "unsupported": self.unsupported_count,
            "is_empty": self.is_empty,
            "failures": [{"path": f.relative_path, "code": f.code} for f in self.failed()],
            "skipped_files": [{"path": f.relative_path, "code": f.code} for f in self.skipped()],
        }


def _resolve_media_type(path: Path, override: str | None) -> str:
    """Decide which loader should handle this file.

    ``.pdf`` is checked before falling back to
    :func:`app.ingestion.loaders.media_type_for_path`, and that ordering is
    load-bearing: that function *raises* for a PDF, because it belongs to
    the text loader module which cannot read them (its message tells callers
    to use ``load_pdf``). This service can read both, so it resolves the
    suffix itself rather than inheriting a guard meant for someone else's
    caller.

    An extension this project does not know resolves to ``text/plain`` — the
    loader's documented ad-hoc behaviour, so a file with no extension still
    works. Discovery is what flags unfamiliar *extensions* as unsupported;
    see :func:`ingest`.
    """
    if override is not None:
        return override
    if path.suffix.lower() == ".pdf":
        return PDF_MEDIA_TYPE
    return media_type_for_path(path)


def _unsupported_reason(media_type: str) -> str:
    """Explain a media type this service cannot read, for a failed result.

    Kept separate from the :class:`~app.domain.errors.ValidationError` the
    same condition raises for a *caller-stated* type, so the two paths can
    never drift apart in wording.
    """
    return (
        f"unsupported media type {media_type!r}; supported types are {sorted(LOADABLE_MEDIA_TYPES)}"
    )


def _load(path: Path, media_type: str) -> TextDocument | PdfDocument:
    """Read one file with the loader that matches its media type.

    Raises
    ------
    ValidationError
        If ``media_type`` is one this project does not handle. Inside
        :func:`ingest_file` this becomes a ``failed`` result, because a file
        with an unhandled *extension* is a data problem rather than a
        programming mistake.
    """
    if can_load_pdf(media_type):
        return load_pdf(path)
    if can_load(media_type):
        return load_text(path, media_type=media_type)
    raise ValidationError(_unsupported_reason(media_type), code="unsupported_media_type")


def _skip_details(document: TextDocument | PdfDocument) -> tuple[str, str]:
    """Return ``(code, reason)`` for a document that has no text.

    The two cases get different codes on purpose: a blank file and a scan
    call for different actions from the user, and a report that lumped them
    together would be actively misleading.
    """
    if isinstance(document, PdfDocument):
        return NO_TEXT_LAYER_CODE, (
            "PDF has no text layer (it appears to be a scan), so there is nothing to index"
        )
    return EMPTY_FILE_CODE, "file contains no text, so there is nothing to index"


def ingest_file(
    path: str | Path,
    *,
    media_type: str | None = None,
    relative_path: str | None = None,
    size_bytes: int | None = None,
) -> FileResult:
    """Load one file, build its metadata, and report what happened.

    This is the single-file entry point. It **never raises for a problem
    with the file itself** — a missing, corrupt, encrypted or unsupported
    file all come back as a ``failed`` :class:`FileResult` carrying the
    typed error's code, so a caller can process a whole directory without
    a try/except per file. It *does* raise when the caller's own arguments
    are wrong, because silently reporting that as a failure would hide a
    programming mistake.

    Parameters
    ----------
    path:
        File to read.
    media_type:
        Override the type derived from the filename. Validated eagerly and
        raises if it is not loadable: the caller *stated* it, so getting it
        wrong is a bug in the caller, not a property of the file.
    relative_path:
        Path to record in the result, relative to whatever root the caller
        was scanning. Defaults to the basename, which is what a direct call
        with no surrounding scan should reasonably report.
    size_bytes:
        Size already known from an earlier ``stat`` (discovery supplies it),
        so the file is not measured twice. Used only when the load fails;
        on success the loaded document's own byte count is authoritative.

    Returns
    -------
    FileResult
        ``loaded``, ``skipped`` or ``failed``.

    Raises
    ------
    ValidationError
        Only for a bad caller-supplied ``media_type``.

    Notes
    -----
    Only :class:`~app.domain.errors.RAGError` is caught. The loaders
    translate filesystem and parsing problems into it, which covers every
    genuine per-file failure; anything else escaping is a defect and is
    allowed to propagate rather than be disguised as a failed file.

    An unfamiliar *extension* is reported as ``unsupported_media_type`` only
    when it arrives through :func:`ingest`, because there discovery
    classifies files before any of them are opened. Called directly with
    such a file, the suffix falls back to ``text/plain`` (the loader's
    documented behaviour for ad-hoc files) and a binary one then fails as
    ``encoding_error`` — which is accurate, though its advice to re-save as
    UTF-8 is aimed at text files.
    """
    file_path = Path(path)
    resolved_type = _resolve_media_type(file_path, media_type)
    recorded_path = relative_path if relative_path is not None else file_path.name

    if media_type is not None and media_type not in LOADABLE_MEDIA_TYPES:
        raise ValidationError(_unsupported_reason(media_type), code="unsupported_media_type")

    try:
        document = _load(file_path, resolved_type)
    except RAGError as exc:
        return FileResult(
            source_name=file_path.name,
            relative_path=recorded_path,
            media_type=resolved_type,
            byte_size=size_bytes,
            outcome=FileOutcome.FAILED,
            code=exc.code,
            reason=exc.message,
        )

    # Built separately from the results so both outcomes share exactly the
    # same metadata, and so each FileResult is constructed with explicit
    # keyword arguments (an unpacked dict of mixed types would defeat the
    # type checker, which is the whole point of it being strict here).
    page_count = document.page_count if isinstance(document, PdfDocument) else None
    pdf_info = document.info if isinstance(document, PdfDocument) else None
    metadata = build_metadata(
        source_name=document.source_name,
        media_type=document.media_type,
        byte_size=document.byte_size,
        text=document.text,
        pdf_info=pdf_info,
        page_count=page_count,
    )

    if document.is_empty:
        code, reason = _skip_details(document)
        return FileResult(
            source_name=document.source_name,
            relative_path=recorded_path,
            media_type=document.media_type,
            byte_size=document.byte_size,
            outcome=FileOutcome.SKIPPED,
            code=code,
            reason=reason,
            document=document,
            metadata=metadata,
            page_count=page_count,
        )

    return FileResult(
        source_name=document.source_name,
        relative_path=recorded_path,
        media_type=document.media_type,
        byte_size=document.byte_size,
        outcome=FileOutcome.LOADED,
        document=document,
        metadata=metadata,
        page_count=page_count,
    )


def ingest(
    root: str | Path,
    *,
    base_dir: str | Path | None = None,
    recursive: bool = True,
) -> IngestResult:
    """Discover every supported file under ``root``, load it, describe it.

    The orchestration the whole ingestion path has been building toward:
    discovery decides *which* files, the loaders decide *what is in them*,
    and metadata decides *how to describe them*.

    Parameters
    ----------
    root:
        Directory to scan.
    base_dir:
        Optional hard boundary — a root outside it raises rather than being
        scanned anyway. Passed straight through to
        :func:`app.ingestion.discovery.discover`, so the containment check
        still happens before anything is opened.
    recursive:
        Whether to descend into subdirectories.

    Returns
    -------
    IngestResult
        One :class:`FileResult` per supported file, plus discovery's
        unsupported list. The result is **always** returned for a valid
        root, even when every file failed — an all-failures batch is
        information, not an exception.

    Raises
    ------
    NotFoundError
        If ``root`` does not exist (or lies outside ``base_dir``).
    ValidationError
        If ``root`` is a file rather than a directory.

    Notes
    -----
    These are *caller* errors — the wrong directory was passed — so they
    propagate instead of being recorded as per-file failures. Once discovery
    succeeds, no single file can raise: :func:`ingest_file` has already
    converted every load error into a result.
    """
    discovery: DiscoveryResult = discover(root, base_dir=base_dir, recursive=recursive)

    results = tuple(
        ingest_file(
            discovered.path,
            media_type=discovered.media_type,
            relative_path=discovered.relative_path,
            size_bytes=discovered.size_bytes,
        )
        for discovered in discovery.files
    )

    return IngestResult(
        root=discovery.root.name,
        files=results,
        unsupported=discovery.unsupported,
    )


def describe(result: IngestResult) -> dict[str, object]:
    """Alias for :meth:`IngestResult.summary` kept for symmetry with
    :func:`app.ingestion.discovery.describe`.

    Both modules expose the same shape of log-safe summary; having the call
    site read ``describe(result)`` regardless of which stage produced the
    result keeps logging code uniform.
    """
    return result.summary()


__all__ = [
    "EMPTY_FILE_CODE",
    "LOADABLE_MEDIA_TYPES",
    "NO_TEXT_LAYER_CODE",
    "FileOutcome",
    "FileResult",
    "IngestResult",
    "describe",
    "ingest",
    "ingest_file",
]
