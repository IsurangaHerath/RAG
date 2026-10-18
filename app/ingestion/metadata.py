"""Document metadata extraction (Stage 13).

Turns the descriptive facts a file already carries into a flat, safe,
storage-ready record. Metadata is what lets a RAG system answer "which
documents mention X?" *before* any text is read, so getting it right early
pays off through every later stage.

What counts as metadata here
----------------------------
* **Embedded** — the PDF document-info dictionary: title, author, subject,
  keywords, the application that produced the file, and its dates.
* **Derived** — facts computed from the extracted text: word count, heading
  count, and the first heading used as a title when the file has no
  embedded one.
* **Provenance** — the source name, media type and byte size, which every
  document has regardless of format.

Three design decisions are worth stating up front, because they are not
obvious and getting them wrong is expensive later.

**Embedded metadata is attacker-controlled.** A PDF's ``/Title`` and
``/Author`` are arbitrary strings chosen by whoever produced the file, and
they flow into logs, API responses and the vector index. pypdf passes
control characters, NUL bytes, newlines and tabs through untouched (a title
of ``"Bad\\x00Title\\x07"`` survives a round-trip verbatim), so every value is
sanitised: control characters stripped, whitespace collapsed, length capped.
This is not paranoia about hypothetical attacks — it is the same reason
:func:`app.ingestion.loaders.load_text` refuses NUL bytes. Untrusted bytes
must not reach a place where they can forge a log line or escape a terminal.

**Absent values are omitted, never ``None``.** ChromaDB rejects metadata
values that are ``None`` or non-scalar, so :meth:`DocumentMetadata.to_mapping`
drops missing fields instead of emitting them. That keeps the same mapping
valid for Chroma, for :class:`app.domain.models.Document`, and for JSON.

**Dates are parsed to timezone-aware UTC, or dropped.** pypdf raises
``ValueError`` on a malformed ``/CreationDate`` — a garbage date in a real
corpus would otherwise crash ingestion. A date we cannot parse is recorded
as missing rather than guessed at, because a fabricated timestamp quietly
corrupts "recently updated" filtering for the life of the document.

Layering note
-------------
This module deliberately imports nothing from the loaders. It consumes
primitive inputs (an open reader, a string, a byte size) and produces
values; :mod:`app.ingestion.pdf_loader` imports *from* it to attach
:class:`PdfInfo` while its reader is already open, which avoids reading the
file twice. Composing a loaded document into a full metadata record is the
ingest service's job, so that composition lives there rather than creating
an import cycle here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone
from enum import StrEnum
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from pypdf import PdfReader

#: PDF document-info keys this module understands, mapped to our field names.
#: Other keys a file may declare (``/Trapped``, custom entries) are ignored
#: rather than copied through: unknown keys are unbounded attacker input with
#: no meaning to us.
_PDF_TEXT_KEYS: Final[dict[str, str]] = {
    "/Title": "title",
    "/Author": "author",
    "/Subject": "subject",
    "/Creator": "creator",
    "/Producer": "producer",
}

#: Keyword separators seen in the wild. PDF stores keywords as one string;
#: producers variously use commas, semicolons or both.
_KEYWORD_SEPARATORS: Final[str] = ",;"

#: Control characters, including NUL and DEL. Tabs and newlines are included
#: because a metadata *value* is a single field — a newline in a title is
#: always wrong and is exactly what log-forging relies on.
_CONTROL_CHARS_RE: Final[re.Pattern[str]] = re.compile(r"[\x00-\x1f\x7f]")

#: Collapse runs of whitespace left behind after control characters go.
_WHITESPACE_RE: Final[re.Pattern[str]] = re.compile(r"\s+")

#: First ATX heading in a Markdown document, e.g. ``# Title``.
_ATX_HEADING_RE: Final[re.Pattern[str]] = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.MULTILINE)

#: An embedded title is a label, not prose: a fixed cap keeps a single
#: hostile document from bloating every record that references it.
MAX_TITLE_LENGTH: Final[int] = 200

#: Other free-text fields are capped too, but more generously, since an
#: abstract or subject line can legitimately be long.
MAX_FIELD_LENGTH: Final[int] = 1000

#: Cap on keyword count so a document cannot mint unlimited filter values.
MAX_KEYWORDS: Final[int] = 32

#: Longest single keyword kept, before the count cap above.
MAX_KEYWORD_LENGTH: Final[int] = 100

#: How many leading lines of a text file are searched for a title. Bounded
#: so a very long preamble cannot become an accidental title.
_TITLE_SCAN_LINES: Final[int] = 20

#: Used when a file offers nothing better. Recorded as such via
#: :attr:`TitleSource.FILENAME`, so it is never mistaken for a real title.
_FALLBACK_TITLE_MAX_LENGTH: Final[int] = 80

#: PDF date strings look like ``D:YYYYMMDDHHmmSSOHH'mm'`` where every part
#: after the year is optional and ``O`` is ``+``, ``-`` or ``Z``. Written as
#: an explicit pattern rather than delegated to pypdf, because the only
#: parser pypdf exposes for this is a private function
#: (``pypdf._utils.parse_iso8824_date``), and building on a private API
#: means a patch release could break ingestion with no warning of ours.
_PDF_DATE_RE: Final[re.Pattern[str]] = re.compile(
    r"^D:(\d{4})(\d{2})?(\d{2})?(\d{2})?(\d{2})?(\d{2})?"
    r"(?:Z|([+\-])(\d{2})?'?(\d{2})?'?)?$"
)


class TitleSource(StrEnum):
    """Where a document's title came from, recorded rather than assumed.

    Provenance is kept because the three sources have very different
    reliability: an embedded title was declared by the author, a heading was
    written into the body, and a filename is usually a machine-generated
    slug. Collapsing them into one string would hide which is which, and a
    downstream ranking or display feature needs to know.
    """

    EMBEDDED = "embedded"
    HEADING = "heading"
    FILENAME = "filename"
    NONE = "none"


@dataclass(frozen=True, slots=True)
class PdfInfo:
    """Sanitised contents of a PDF document-info dictionary.

    Every string here has already been cleaned and bounded, so this type is
    safe to log, serialise and store as-is. Absent fields are ``None``.
    """

    title: str | None
    author: str | None
    subject: str | None
    keywords: tuple[str, ...]
    creator: str | None
    producer: str | None
    created_at: datetime | None
    modified_at: datetime | None

    @property
    def is_empty(self) -> bool:
        """True when the file declared no usable metadata at all.

        Not an error: plenty of valid PDFs carry an empty info dictionary,
        and the ingest service treats this as "nothing to record" rather
        than a failure.
        """
        return not (
            self.title
            or self.author
            or self.subject
            or self.keywords
            or self.creator
            or self.created_at
            or self.modified_at
        )


@dataclass(frozen=True, slots=True)
class TextInfo:
    """Facts computed from extracted text rather than declared by the file."""

    heading_count: int
    word_count: int
    line_count: int
    char_count: int


@dataclass(frozen=True, slots=True)
class DocumentMetadata:
    """Flat, sanitised, storage-ready metadata for one document.

    Immutable, like every model in this project, so a record cannot be
    mutated out from under a worker that is still reading it.

    Parameters
    ----------
    source_name:
        Basename only — never a path, and never an absolute one.
    byte_size:
        Size of the source file on disk.
    title / author / subject / creator / producer:
        Sanitised embedded values, or values derived from the text where the
        file declared none. See :attr:`title_source` for how the title was
        obtained.
    keywords:
        Individual keywords, already split out of the single
        comma-separated string PDF stores them as.
    created_at / modified_at:
        Timezone-aware UTC. ``None`` when absent *or* unparseable.
    page_count:
        Pages for paged formats; ``None`` for plain text.
    word_count / heading_count / char_count:
        Derived from the extracted text.
    """

    source_name: str
    media_type: str
    byte_size: int
    title: str | None = None
    title_source: TitleSource = TitleSource.NONE
    author: str | None = None
    subject: str | None = None
    keywords: tuple[str, ...] = ()
    creator: str | None = None
    producer: str | None = None
    created_at: datetime | None = None
    modified_at: datetime | None = None
    page_count: int | None = None
    word_count: int = 0
    heading_count: int = 0
    char_count: int = 0

    @property
    def is_empty(self) -> bool:
        """True when the document itself declared no metadata.

        A title derived from the filename does **not** count, even though
        :attr:`title` is then non-``None``: that slug is something we
        invented for display, not a fact the file supplied. Treating it as
        real would make a completely metadata-free PDF report
        ``is_empty == False``, hiding the very case this flag exists to
        surface.
        """
        declared_title = self.title is not None and self.title_source is not TitleSource.FILENAME
        return (
            not declared_title
            and self.author is None
            and self.subject is None
            and not self.keywords
            and self.creator is None
            and self.created_at is None
            and self.modified_at is None
        )

    def to_mapping(self) -> dict[str, str | int | float | bool]:
        """Return a flat mapping safe for ChromaDB, JSON and logs.

        Absent values are **omitted rather than set to ``None``**, because
        ChromaDB rejects ``None`` metadata and a dict containing one would
        fail at index time — long after the code that produced it looked
        fine. Datetimes become ISO-8601 strings and keywords are joined into
        one string, since neither store accepts datetime or list values.
        """
        mapping: dict[str, str | int | float | bool] = {
            "source_name": self.source_name,
            "media_type": self.media_type,
            "byte_size": self.byte_size,
            "word_count": self.word_count,
            "heading_count": self.heading_count,
            "char_count": self.char_count,
        }

        optional: dict[str, str | int | float | bool | None] = {
            "title": self.title,
            "title_source": str(self.title_source),
            "author": self.author,
            "subject": self.subject,
            "keywords": ", ".join(self.keywords) if self.keywords else None,
            "creator": self.creator,
            "producer": self.producer,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "modified_at": self.modified_at.isoformat() if self.modified_at else None,
            "page_count": self.page_count,
        }
        mapping.update({k: v for k, v in optional.items() if v is not None})
        return mapping

    def summary(self) -> dict[str, str | int | float | bool | list[str] | None]:
        """JSON-safe summary for logs and API responses.

        Wider than :meth:`to_mapping`: it always reports every field, using
        ``None`` for absent ones, so a log line shows what was *missing* as
        well as what was found. Never contains document text.
        """
        # Built field by field rather than by spreading to_mapping(), which
        # deliberately drops absent values. A log line that hides what was
        # missing is far less useful than one that shows it as None.
        return {
            "source_name": self.source_name,
            "media_type": self.media_type,
            "byte_size": self.byte_size,
            "title": self.title,
            "title_source": str(self.title_source),
            "author": self.author,
            "subject": self.subject,
            "keywords": list(self.keywords),
            "creator": self.creator,
            "producer": self.producer,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "modified_at": self.modified_at.isoformat() if self.modified_at else None,
            "page_count": self.page_count,
            "word_count": self.word_count,
            "heading_count": self.heading_count,
            "char_count": self.char_count,
        }


def sanitize_text(
    value: object,
    *,
    max_length: int = MAX_FIELD_LENGTH,
) -> str | None:
    """Coerce an untrusted metadata value into a safe, bounded string.

    Accepts the shapes producers actually emit: ``str`` (including pypdf's
    ``TextStringObject``), and numbers, which arrive already stringified as
    ``"42"``. Control characters are stripped, whitespace runs collapsed,
    and the result truncated to ``max_length``.

    Returns
    -------
    str | None
        The cleaned value, or ``None`` when nothing usable was left — an
        empty or whitespace-only value becomes ``None`` rather than ``""``,
        so callers can tell "absent" from "present but blank" without a
        second check.

    Notes
    -----
    Nothing is ASCII-folded. Non-ASCII titles (Sinhala, emoji, CJK) are
    legitimate and round-trip correctly, so folding them to ``?`` would
    destroy real information while appearing to tidy up.
    """
    if value is None:
        return None
    if not isinstance(value, (str, int, float, bool)):
        return None

    cleaned = _CONTROL_CHARS_RE.sub(" ", str(value))
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()

    if not cleaned:
        return None
    return cleaned[:max_length]


def sanitize_keywords(value: object, *, limit: int = MAX_KEYWORDS) -> tuple[str, ...]:
    """Split a PDF keyword string into individual, cleaned keywords.

    PDF stores keywords as a single string, so a filter on one keyword needs
    them separated first. Duplicates are removed while keeping first-seen
    order, so results stay deterministic across runs.
    """
    if not isinstance(value, str) or not value.strip():
        return ()

    # rf-string on purpose: a plain f-string would make "\s" an invalid
    # escape sequence and emit a SyntaxWarning, which pytest escalates to an
    # error via filterwarnings = ["error"].
    parts = re.split(rf"[{_KEYWORD_SEPARATORS}\s]+", value)
    seen: set[str] = set()
    keywords: list[str] = []

    for part in parts:
        cleaned = sanitize_text(part, max_length=MAX_KEYWORD_LENGTH)
        if cleaned is None or cleaned in seen:
            continue
        seen.add(cleaned)
        keywords.append(cleaned)
        if len(keywords) >= limit:
            break

    return tuple(keywords)


def parse_pdf_datetime(value: object) -> datetime | None:
    """Parse a PDF date into timezone-aware UTC, or return ``None``.

    Two safety properties, both learned from real files:

    * pypdf raises ``ValueError`` on a malformed date such as
      ``"not-a-date"``. A corpus will contain one of these eventually, and
      ingestion must not die on it, so the failure is caught and reported as
      "no date" rather than propagated.
    * A naive datetime is interpreted as UTC rather than left naive. The
      domain model rejects naive datetimes outright, because the same
      ambiguous timestamp means different moments on different machines.
    """
    if value is None:
        return None

    # Annotated explicitly: without it mypy infers `datetime` from the first
    # branch and then rejects the str branch, which legitimately yields None.
    parsed: datetime | None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        parsed = _parse_pdf_date_string(value)
    else:
        return None

    if parsed is None:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _parse_pdf_date_string(value: str) -> datetime | None:
    """Parse a ``D:YYYYMMDDHHmmSSOHH'mm'`` string, or return ``None``.

    Every component after the year is optional, as the PDF spec allows, so a
    bare ``D:2026`` is valid and means 1 January 2026. Unparseable input
    returns ``None`` rather than raising: a document with a corrupt date is
    still a document, and losing the whole file over one bad field would be
    a poor trade.

    Written against the documented format rather than imported from pypdf,
    whose only exposed parser for this is private
    (``pypdf._utils.parse_iso8824_date``).
    """
    match = _PDF_DATE_RE.match(value.strip())
    if match is None:
        return None

    year = int(match.group(1))
    month = int(match.group(2) or 1)
    day = int(match.group(3) or 1)
    hour = int(match.group(4) or 0)
    minute = int(match.group(5) or 0)
    second = int(match.group(6) or 0)

    sign = match.group(7)
    offset_hours = int(match.group(8) or 0)
    offset_minutes = int(match.group(9) or 0)

    try:
        base = datetime(year, month, day, hour, minute, second)
    except ValueError:
        # e.g. month 13 or day 32: syntactically fine, semantically impossible.
        return None

    if sign is None or sign == "Z":
        return base.replace(tzinfo=UTC)

    offset = timedelta(hours=offset_hours, minutes=offset_minutes)
    if sign == "-":
        offset = -offset
    return base.replace(tzinfo=timezone(offset))


def _read_pdf_date(info: object, attribute: str) -> datetime | None:
    """Read one date via pypdf's parsed property, tolerating bad input.

    Uses the property rather than the raw string so pypdf's own format
    handling stays in one place; wraps it because the property raises on
    malformed values.
    """
    try:
        value = getattr(info, attribute, None)
    except Exception:
        return None
    return parse_pdf_datetime(value)


def extract_pdf_info(reader: PdfReader) -> PdfInfo:
    """Read and sanitise a PDF's document-info dictionary.

    Takes an *open* reader rather than a path, so the PDF loader can pass the
    reader it already has instead of opening the file a second time.

    Parameters
    ----------
    reader:
        A pypdf reader whose pages have already been made accessible (call
        this after decryption).

    Returns
    -------
    PdfInfo
        Sanitised values. A PDF with no info dictionary at all — which
        :attr:`pypdf.PdfReader.metadata` reports as ``None`` — yields an
        empty but valid ``PdfInfo``.

    Notes
    -----
    ``/Producer`` defaults to ``"pypdf"`` in files that never set it, so it
    is frequently the library that wrote the file rather than anything about
    the document. It is recorded for completeness but should not be
    presented to a user as meaningful.
    """
    try:
        info = reader.metadata
    except Exception:
        return PdfInfo(
            title=None,
            author=None,
            subject=None,
            keywords=(),
            creator=None,
            producer=None,
            created_at=None,
            modified_at=None,
        )

    if info is None:
        return PdfInfo(
            title=None,
            author=None,
            subject=None,
            keywords=(),
            creator=None,
            producer=None,
            created_at=None,
            modified_at=None,
        )

    values: dict[str, str | None] = {}
    for key, field_name in _PDF_TEXT_KEYS.items():
        try:
            raw = info.get(key, None)
        except Exception:
            raw = None
        values[field_name] = sanitize_text(
            raw, max_length=MAX_TITLE_LENGTH if field_name == "title" else MAX_FIELD_LENGTH
        )

    try:
        raw_keywords = info.get("/Keywords", None)
    except Exception:
        raw_keywords = None

    return PdfInfo(
        title=values["title"],
        author=values["author"],
        subject=values["subject"],
        keywords=sanitize_keywords(raw_keywords),
        creator=values["creator"],
        producer=values["producer"],
        created_at=_read_pdf_date(info, "creation_date"),
        modified_at=_read_pdf_date(info, "modification_date"),
    )


def extract_text_info(text: str) -> TextInfo:
    """Count the structural facts a chunker will want later.

    Heading count matters because Markdown headings are where section
    boundaries live, and Stage 18 splits on them; recording the count now
    costs nothing and makes "why did this document chunk badly?" answerable.
    """
    heading_count = len(_ATX_HEADING_RE.findall(text))
    return TextInfo(
        heading_count=heading_count,
        word_count=len(text.split()),
        line_count=text.count("\n") + 1 if text else 0,
        char_count=len(text),
    )


def extract_title_from_text(text: str) -> str | None:
    """Return the first Markdown heading, or the first non-empty line.

    A heading is preferred because it is the document naming itself. Falling
    back to the first line covers plain-text files, which usually start with
    a title even though they have no Markdown syntax.

    Both steps inspect only the leading lines. Without that bound, a heading
    buried hundreds of lines down would silently become the document's
    title — the search would also cost a full scan of every file.
    """
    # Slice BEFORE matching: a regex search over the whole text would find a
    # heading anywhere in the document, not just in the title position.
    head = "\n".join(text.splitlines()[:_TITLE_SCAN_LINES])

    match = _ATX_HEADING_RE.search(head)
    if match:
        return sanitize_text(match.group(1), max_length=MAX_TITLE_LENGTH)

    for line in head.splitlines():
        cleaned = sanitize_text(line, max_length=MAX_TITLE_LENGTH)
        if cleaned is not None:
            return cleaned
    return None


def fallback_title(source_name: str) -> str | None:
    """Derive a readable title from a filename, as a last resort.

    ``payment-gateway-spec.md`` becomes ``Payment gateway spec``. Only used
    when the document offers nothing else, and always recorded as
    :attr:`TitleSource.FILENAME` so nothing downstream mistakes it for a
    title the author wrote.
    """
    stem = source_name.rsplit("/", 1)[-1]
    if "." in stem:
        stem = stem.rsplit(".", 1)[0]
    cleaned = sanitize_text(stem.replace("-", " ").replace("_", " "), max_length=80)
    if cleaned is None:
        return None
    return cleaned[:_FALLBACK_TITLE_MAX_LENGTH].capitalize()


def build_metadata(
    *,
    source_name: str,
    media_type: str,
    byte_size: int,
    text: str = "",
    pdf_info: PdfInfo | None = None,
    page_count: int | None = None,
) -> DocumentMetadata:
    """Combine embedded, derived and provenance facts into one record.

    This is the composition point for a single document. Title resolution
    runs embedded-first, then first-heading, then filename, and
    :attr:`title_source` records which one won. Preferring the embedded
    title is deliberate: it is what the author declared, whereas a heading
    is only a guess and a filename is often meaningless.

    Parameters
    ----------
    source_name:
        Basename of the file, e.g. ``api-spec.md``.
    media_type:
        MIME type, e.g. ``text/markdown``.
    byte_size:
        Size of the file on disk.
    text:
        Extracted document text, used for the derived counts and the
        heading-based title fallback.
    pdf_info:
        Already-extracted PDF info, when the document is a PDF.
    page_count:
        Page count for paged formats; ``None`` for plain text.
    """
    info = extract_text_info(text)

    title: str | None = None
    title_source = TitleSource.NONE

    if pdf_info is not None and pdf_info.title:
        title, title_source = pdf_info.title, TitleSource.EMBEDDED

    if title is None:
        derived = extract_title_from_text(text)
        if derived:
            title, title_source = derived, TitleSource.HEADING

    if title is None:
        derived = fallback_title(source_name)
        if derived:
            title, title_source = derived, TitleSource.FILENAME

    return DocumentMetadata(
        source_name=source_name,
        media_type=media_type,
        byte_size=byte_size,
        title=title,
        title_source=title_source,
        author=pdf_info.author if pdf_info else None,
        subject=pdf_info.subject if pdf_info else None,
        keywords=pdf_info.keywords if pdf_info else (),
        creator=pdf_info.creator if pdf_info else None,
        producer=pdf_info.producer if pdf_info else None,
        created_at=pdf_info.created_at if pdf_info else None,
        modified_at=pdf_info.modified_at if pdf_info else None,
        page_count=page_count,
        word_count=info.word_count,
        heading_count=info.heading_count,
        char_count=info.char_count,
    )


__all__ = [
    "MAX_FIELD_LENGTH",
    "MAX_KEYWORDS",
    "MAX_TITLE_LENGTH",
    "DocumentMetadata",
    "PdfInfo",
    "TextInfo",
    "TitleSource",
    "build_metadata",
    "extract_pdf_info",
    "extract_text_info",
    "extract_title_from_text",
    "fallback_title",
    "parse_pdf_datetime",
    "sanitize_keywords",
    "sanitize_text",
]
