"""Core domain models shared by every package (Stage 09).

These types are deliberately framework-free: no FastAPI, no LangChain, no
pydantic.  ``app/domain`` is the innermost layer, so it must not import from
anything else in the application.  Keeping it pure means the models can be
used by the ingestion worker, the API, the CLI and the test suite alike.

Design note on identifiers
--------------------------
Two independent identifiers exist, and mixing them up causes real bugs:

``doc_id``
    Derived from the *source path*, e.g. ``doc_9f2c1ab34de5``.  It is
    stable across content edits, so re-ingesting a file you updated still
    targets the same logical document and its stale chunks can be replaced.

``content_sha256``
    Derived from the *bytes*.  It changes whenever the file is edited and
    stays the same when a file is merely renamed or copied.

That split is what lets Stage 15 answer "is this file new, edited, renamed
or unchanged?" without guessing.  If ``doc_id`` were content-derived, an
edited file would look like a brand-new document and its old chunks would
never be cleaned up.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from app.domain.errors import ValidationError

# --- Supported media types -----------------------------------------------------

#: What the ingestion pipeline can actually load today (Stages 11-12).
#: The *model* only requires a syntactically valid MIME type; whether a type
#: is supported is a discovery-layer policy, so this stays a hint here.
SUPPORTED_MEDIA_TYPES: frozenset[str] = frozenset(
    {
        "application/pdf",
        "text/plain",
        "text/markdown",
    }
)

#: Extension -> MIME type, used by file discovery (Stage 10).
EXTENSION_TO_MEDIA_TYPE: dict[str, str] = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".md": "text/markdown",
}

# --- Validation constants ------------------------------------------------------

_DOC_ID_RE = re.compile(r"^doc_[0-9a-f]{16}$")
_MIME_RE = re.compile(r"^[a-z0-9][a-z0-9!#$&^_.+-]*/[a-z0-9][a-z0-9!#$&^_.+-]*$")
# Control characters (including NUL) never belong in a filename we display.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f]")

_ID_PREFIX = "doc"
_ID_HEX_LENGTH = 16  # 64 bits: ample for document-scale corpora, short to read


class IngestStatus(StrEnum):
    """Lifecycle of a document as it moves through ingestion.

    ``StrEnum`` so the value serialises straight to JSON and compares equal
    to its string form (``record.status == "indexed"`` works).
    """

    PENDING = "pending"
    PARSING = "parsing"
    INDEXING = "indexing"
    INDEXED = "indexed"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        """True once processing can no longer proceed on its own."""
        return self in {IngestStatus.INDEXED, IngestStatus.FAILED}


# --- Stable identifier helpers -------------------------------------------------


def make_doc_id(source_path: str) -> str:
    """Build a stable ``doc_id`` from a document's source path.

    The path is normalised first (``\\`` -> ``/``, leading ``./`` stripped)
    so the same file produces the same id on Windows and POSIX, and whether
    it is opened as ``data/a.md`` or ``./data/a.md``.

    Parameters
    ----------
    source_path:
        Path relative to the configured data directory. Absolute paths are
        rejected: they leak local filesystem layout into stored records.

    Returns
    -------
    str
        ``doc_`` followed by 16 lowercase hex characters.

    Raises
    ------
    ValidationError
        If the path is empty, absolute, or contains traversal.
    """
    if not isinstance(source_path, str) or not source_path.strip():
        raise ValidationError(
            "source path must be a non-empty string",
            code="empty_source_path",
        )

    cleaned = source_path.strip().replace("\\", "/")

    if cleaned.startswith("/") or re.match(r"^[a-zA-Z]:/", cleaned):
        raise ValidationError(
            "source path must be relative to the data directory, not absolute",
            code="absolute_source_path",
        )

    if any(part == ".." for part in cleaned.split("/")):
        raise ValidationError(
            "source path must not contain '..' traversal segments",
            code="path_traversal",
        )

    while cleaned.startswith("./"):
        cleaned = cleaned[2:]

    if not cleaned:
        raise ValidationError(
            "source path must not resolve to an empty path",
            code="empty_source_path",
        )

    digest = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[:_ID_HEX_LENGTH]
    return f"{_ID_PREFIX}_{digest}"


def make_chunk_id(doc_id: str, index: int) -> str:
    """Build a stable chunk id such as ``doc_9f2c1ab34de5:7``.

    Stable for unchanged input and unchanged settings, which is what makes
    re-ingestion idempotent (Stage 23): the same chunk is overwritten rather
    than appended a second time.
    """
    if index < 0:
        raise ValidationError(f"chunk index must be >= 0, got {index}", code="negative_chunk_index")
    return f"{doc_id}:{index}"


def content_sha256(content: bytes | str) -> str:
    """Return the lowercase hex SHA-256 of the given content.

    Accepts ``bytes`` or ``str`` (encoded as UTF-8) so callers can hash a
    file on disk or already-decoded text interchangeably.
    """
    return _sha256_hex(content)


def _sha256_hex(content: bytes | str) -> str:
    """Internal implementation shared by :func:`content_sha256`.

    Exists as a separate name so a parameter or local variable called
    ``content_sha256`` cannot shadow it and break hashing.
    """
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.sha256(content).hexdigest()


# --- The model ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Document:
    """An ingested document and its processing state.

    Immutable (``frozen``): state changes go through :meth:`with_status`,
    which returns a new instance.  That prevents a worker from mutating a
    record another task is still reading.

    Parameters
    ----------
    doc_id:
        Stable id from :func:`make_doc_id`; see module docstring.
    filename:
        Basename as displayed to users, e.g. ``api-spec.md``. Never a path.
    media_type:
        MIME type such as ``text/markdown``.
    content:
        Full extracted text. May be empty for an unreadable/blank file —
        loaders (Stage 11) must be able to represent that without crashing;
        the ingestion service decides whether to skip it.
    metadata:
        Free-form extras (category, author, user tags). Values are limited
        to JSON scalars so the record stays serialisable everywhere.
    created_at:
        When the source file was created/first seen. Timezone-aware.
    status:
        Current :class:`IngestStatus`.
    content_sha256:
        Hash of the bytes; see module docstring.
    source_path:
        Path relative to the data directory. Never absolute.
    page_count:
        Total pages for paged formats (PDF); ``None`` for plain text.
    error:
        Failure reason; required when ``status`` is ``FAILED``.
    ingested_at:
        When ingestion completed.
    """

    doc_id: str
    filename: str
    media_type: str
    content: str
    metadata: Mapping[str, str | int | float | bool]
    created_at: datetime
    status: IngestStatus = IngestStatus.PENDING
    content_sha256: str = ""
    source_path: str = ""
    page_count: int | None = None
    error: str | None = None
    ingested_at: datetime | None = None

    # -- construction -------------------------------------------------------

    @classmethod
    def create(
        cls,
        *,
        source_path: str,
        filename: str,
        media_type: str,
        content: str,
        created_at: datetime | None = None,
        metadata: Mapping[str, str | int | float | bool] | None = None,
        content_hash: str | None = None,
        page_count: int | None = None,
    ) -> Document:
        """Build a validated document, deriving ``doc_id`` and the hash.

        This is the normal entry point: it guarantees the two identifiers
        are always consistent with the values they are derived from, which
        manually assembling the dataclass would not.

        ``content_hash`` is an optional escape hatch for callers that already
        computed the digest (e.g. a file-level hash from the discovery pass).
        It is deliberately *not* named ``content_sha256`` — a parameter with
        that name would shadow the module-level function of the same name.
        """
        # Validate BEFORE coercing: dict(["a","b"]) raises a raw ValueError
        # from inside dict(), which would escape as an untyped error instead
        # of our ValidationError with its machine-readable code.
        if metadata is not None and not isinstance(metadata, Mapping):
            raise ValidationError(
                f"metadata must be a mapping, got {type(metadata).__name__}",
                code="invalid_metadata",
            )

        digest = content_hash if content_hash is not None else _sha256_hex(content)
        return cls(
            doc_id=make_doc_id(source_path),
            filename=filename,
            media_type=media_type,
            content=content,
            metadata=dict(metadata or {}),
            created_at=created_at or datetime.now(UTC),
            status=IngestStatus.PENDING,
            content_sha256=digest,
            source_path=source_path.replace("\\", "/"),
            page_count=page_count,
        )

    # -- validation ---------------------------------------------------------

    def __post_init__(self) -> None:
        """Validate every field once, at construction time."""
        # Canonicalise separators FIRST, before any validation, so that
        # `Document(...)` and `Document.create(...)` behave identically.
        # Without this, the same Windows path could yield two different
        # source_path values depending on the entry point used.
        if isinstance(self.source_path, str) and "\\" in self.source_path:
            object.__setattr__(self, "source_path", self.source_path.replace("\\", "/"))

        self._require(self.doc_id, "doc_id")
        if not _DOC_ID_RE.match(self.doc_id):
            raise ValidationError(
                f"doc_id must look like 'doc_' + 16 hex chars, got {self.doc_id!r}",
                code="invalid_doc_id",
            )

        self._require(self.filename, "filename")
        if any(sep in self.filename for sep in ("/", "\\")):
            raise ValidationError(
                "filename must be a basename without directory separators; "
                "store directories in source_path instead",
                code="filename_contains_path",
            )
        if self.filename in {".", ".."}:
            raise ValidationError(
                f"filename {self.filename!r} is not a valid file name",
                code="invalid_filename",
            )
        if _CONTROL_CHARS_RE.search(self.filename):
            raise ValidationError(
                "filename must not contain control characters",
                code="invalid_filename",
            )

        self._require(self.media_type, "media_type")
        if not _MIME_RE.match(self.media_type):
            raise ValidationError(
                f"media_type must be a valid MIME type like 'text/markdown', "
                f"got {self.media_type!r}",
                code="invalid_media_type",
            )

        if not isinstance(self.content, str):
            raise ValidationError(
                f"content must be str, got {type(self.content).__name__}",
                code="invalid_content",
            )

        self._validate_metadata(self.metadata)

        if not isinstance(self.created_at, datetime):
            raise ValidationError("created_at must be a datetime", code="invalid_created_at")
        if self.created_at.tzinfo is None:
            raise ValidationError(
                "created_at must be timezone-aware (naive datetimes are ambiguous "
                "across machines and containers)",
                code="naive_datetime",
            )

        if self.ingested_at is not None and self.ingested_at.tzinfo is None:
            raise ValidationError("ingested_at must be timezone-aware", code="naive_datetime")

        if self.page_count is not None and self.page_count < 0:
            raise ValidationError(
                f"page_count must be >= 0, got {self.page_count}",
                code="negative_page_count",
            )

        if not isinstance(self.status, IngestStatus):
            raise ValidationError(
                f"status must be an IngestStatus, got {self.status!r}",
                code="invalid_status",
            )

        # Cross-field rule: a failure with no reason is undebuggable, and a
        # reason without a failure means stale text on a healthy record.
        if self.status is IngestStatus.FAILED and not (self.error or "").strip():
            raise ValidationError(
                "status is 'failed' but no error message was recorded",
                code="missing_error",
            )
        if self.status is not IngestStatus.FAILED and self.error:
            raise ValidationError(
                f"error may only be set when status is 'failed', not {self.status!r}",
                code="unexpected_error",
            )

        if self.source_path and (
            self.source_path.startswith("/") or re.match(r"^[a-zA-Z]:/", self.source_path)
        ):
            raise ValidationError(
                "source_path must be relative to the data directory",
                code="absolute_source_path",
            )

        if self.status is IngestStatus.INDEXED and self.ingested_at is None:
            raise ValidationError(
                "an indexed document must record when it was ingested",
                code="missing_ingested_at",
            )

    @staticmethod
    def _require(value: object, name: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValidationError(f"{name} must be a non-empty string", code=f"empty_{name}")

    @staticmethod
    def _validate_metadata(metadata: Mapping[str, str | int | float | bool]) -> None:
        """Metadata must be a mapping of non-empty keys to JSON scalars.

        Rejecting nested structures and ``None`` keeps the record safe to
        store in ChromaDB, send as JSON, and log — all of which choke on
        arbitrary object graphs.
        """
        if not isinstance(metadata, Mapping):
            raise ValidationError(
                f"metadata must be a mapping, got {type(metadata).__name__}",
                code="invalid_metadata",
            )

        for key, value in metadata.items():
            if not isinstance(key, str) or not key.strip():
                raise ValidationError(
                    "metadata keys must be non-empty strings",
                    code="invalid_metadata_key",
                )
            # bool is checked before int on purpose: bool subclasses int, so
            # both are allowed, but the ordering keeps the intent readable.
            if not isinstance(value, (str, int, float, bool)):
                raise ValidationError(
                    f"metadata[{key!r}] must be a string, number or bool — "
                    f"got {type(value).__name__}. Nested structures and None are "
                    f"not supported because they cannot be stored in the vector "
                    f"index.",
                    code="invalid_metadata_value",
                )

    # -- derived helpers ----------------------------------------------------

    @property
    def char_count(self) -> int:
        """Number of characters of extracted text (0 for an empty document)."""
        return len(self.content)

    @property
    def is_empty(self) -> bool:
        """True when no text could be extracted.

        The ingestion service (Stage 14) uses this to mark a file *skipped*
        rather than failing the whole batch — e.g. a scanned PDF with no
        text layer, which we cannot read without OCR.
        """
        return not self.content.strip()

    @property
    def is_supported(self) -> bool:
        """Whether a loader exists for this media type."""
        return self.media_type in SUPPORTED_MEDIA_TYPES

    @property
    def is_terminal(self) -> bool:
        """Whether processing has finished (successfully or not)."""
        return self.status.is_terminal

    def with_status(
        self,
        status: IngestStatus,
        *,
        error: str | None = None,
        ingested_at: datetime | None = None,
    ) -> Document:
        """Return a copy in a new status, validating the transition.

        Provided because ``frozen=True`` forbids assignment; this is the one
        sanctioned way to move a document through its lifecycle.

        Raises
        ------
        ValidationError
            If the transition is not allowed (see :meth:`allowed_transitions`).
        """
        if status not in self.allowed_transitions():
            raise ValidationError(
                f"cannot move document {self.doc_id} from {self.status!r} to {status!r}",
                code="invalid_status_transition",
            )

        if status is IngestStatus.FAILED and not (error or "").strip():
            raise ValidationError(
                "a transition to 'failed' must include an error message",
                code="missing_error",
            )

        timestamp = ingested_at if ingested_at is not None else datetime.now(UTC)
        return replace(
            self,
            status=status,
            error=error if status is IngestStatus.FAILED else None,
            ingested_at=timestamp if status is IngestStatus.INDEXED else self.ingested_at,
        )

    def allowed_transitions(self) -> frozenset[IngestStatus]:
        """Return the statuses this document may move to from its current one.

        Terminal states allow nothing: a finished document must be re-ingested
        (a fresh record) rather than nudged back into ``PENDING``, so a stale
        index can never be silently resurrected.
        """
        if self.status.is_terminal:
            return frozenset()

        graph: dict[IngestStatus, set[IngestStatus]] = {
            IngestStatus.PENDING: {IngestStatus.PARSING, IngestStatus.FAILED},
            IngestStatus.PARSING: {IngestStatus.INDEXING, IngestStatus.FAILED},
            IngestStatus.INDEXING: {IngestStatus.INDEXED, IngestStatus.FAILED},
        }
        return frozenset(graph.get(self.status, set()))

    def summary(self) -> dict[str, Any]:
        """Return a JSON-safe summary for API responses and logs.

        Deliberately excludes ``content``: document text must never land in
        a log line or an API payload that does not need it.
        """
        return {
            "doc_id": self.doc_id,
            "filename": self.filename,
            "media_type": self.media_type,
            "status": str(self.status),
            "char_count": self.char_count,
            "page_count": self.page_count,
            "source_path": self.source_path,
            "created_at": self.created_at.isoformat(),
            "ingested_at": self.ingested_at.isoformat() if self.ingested_at else None,
            "error": self.error,
            "metadata": dict(self.metadata),
        }


__all__ = [
    "EXTENSION_TO_MEDIA_TYPE",
    "SUPPORTED_MEDIA_TYPES",
    "Document",
    "IngestStatus",
    "content_sha256",
    "make_chunk_id",
    "make_doc_id",
]
