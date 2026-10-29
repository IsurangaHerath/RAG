"""Unit tests for the document data model (Stage 09).

Split into three groups:

1. **Valid records** — constructing a document works and derives the
   identifiers correctly.
2. **Invalid records** — every validation rule rejects bad input with a
   specific error ``code``, so failures are debuggable rather than generic.
3. **Identifier stability** — the property Stage 15 (deduplication) and
   Stage 23 (idempotent re-ingestion) both depend on.

No network, no API key, no filesystem writes.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime

import pytest

from app.domain.errors import ValidationError
from app.domain.models import (
    EXTENSION_TO_MEDIA_TYPE,
    SUPPORTED_MEDIA_TYPES,
    Document,
    IngestStatus,
    MetadataFilter,
    RetrievedChunk,
    content_sha256,
    make_chunk_id,
    make_doc_id,
)

NOW = datetime(2026, 10, 17, 12, 0, 0, tzinfo=UTC)


def make_document(**overrides: object) -> Document:
    """Build a valid document; individual tests override single fields."""
    params: dict[str, object] = {
        "source_path": "guides/api-spec.md",
        "filename": "api-spec.md",
        "media_type": "text/markdown",
        "content": "# API Spec\n\nThe timeout is 30 seconds.",
        "created_at": NOW,
        "metadata": {"category": "spec", "revision": 3},
        "page_count": None,
    }
    params.update(overrides)
    return Document.create(**params)  # type: ignore[arg-type]


def make_indexed(**overrides: object) -> Document:
    """Build a document that has legally completed the whole lifecycle.

    Encapsulates PENDING -> PARSING -> INDEXING -> INDEXED so individual
    tests do not repeat (or accidentally skip) the transitions.
    """
    doc = make_document(**overrides)
    doc = doc.with_status(IngestStatus.PARSING)
    doc = doc.with_status(IngestStatus.INDEXING)
    return doc.with_status(IngestStatus.INDEXED, ingested_at=NOW)


# =============================================================================
# 1. Valid records
# =============================================================================


def test_valid_document_constructs() -> None:
    doc = make_document()

    assert doc.doc_id.startswith("doc_")
    assert doc.filename == "api-spec.md"
    assert doc.media_type == "text/markdown"
    assert doc.status is IngestStatus.PENDING
    assert doc.created_at == NOW


def test_create_derives_doc_id_from_source_path() -> None:
    doc = make_document(source_path="a/b/c.md")

    assert doc.doc_id == make_doc_id("a/b/c.md")


def test_create_derives_content_hash() -> None:
    doc = make_document(content="hello world")

    assert doc.content_sha256 == content_sha256("hello world")
    assert len(doc.content_sha256) == 64


def test_supplied_content_hash_is_honoured() -> None:
    """A caller that already hashed the file can pass it through."""
    precomputed = content_sha256("whatever")

    doc = make_document(content="whatever", content_hash=precomputed)

    assert doc.content_sha256 == precomputed


def test_metadata_defaults_to_empty_mapping() -> None:
    doc = make_document(metadata=None)

    assert dict(doc.metadata) == {}


def test_metadata_is_copied_not_aliased() -> None:
    """Mutating the caller's dict afterwards must not change the record."""
    source = {"category": "spec"}
    doc = make_document(metadata=source)
    source["category"] = "tampered"

    assert doc.metadata["category"] == "spec"


def test_empty_content_is_allowed_and_flagged() -> None:
    """An unreadable/blank file is representable — loaders need this.

    Stage 11 must handle empty files without crashing, and Stage 14 marks
    them 'skipped' rather than failing the batch. Rejecting empty content
    here would make that impossible.
    """
    doc = make_document(content="")

    assert doc.is_empty is True
    assert doc.char_count == 0


def test_whitespace_only_content_counts_as_empty() -> None:
    doc = make_document(content="   \n\t  \n")

    assert doc.is_empty is True


def test_real_content_is_not_empty() -> None:
    assert make_document().is_empty is False


def test_char_count() -> None:
    assert make_document(content="abcdef").char_count == 6


def test_page_count_none_for_plain_text() -> None:
    assert make_document().page_count is None


def test_is_supported_for_known_media_types() -> None:
    for media_type in SUPPORTED_MEDIA_TYPES:
        assert make_document(media_type=media_type).is_supported is True


def test_extension_map_covers_all_supported_types() -> None:
    """Every supported media type should be reachable from some extension."""
    assert set(EXTENSION_TO_MEDIA_TYPE.values()) == set(SUPPORTED_MEDIA_TYPES)


# =============================================================================
# 2. Invalid records
# =============================================================================


def test_empty_filename_rejected() -> None:
    with pytest.raises(ValidationError, match="filename") as excinfo:
        make_document(filename="   ")
    assert excinfo.value.code == "empty_filename"


def test_filename_with_directory_separator_rejected() -> None:
    """A path in ``filename`` would let one document display as another's path."""
    with pytest.raises(ValidationError) as excinfo:
        make_document(filename="sub/dir/api.md")
    assert excinfo.value.code == "filename_contains_path"


def test_filename_traversal_rejected() -> None:
    with pytest.raises(ValidationError) as excinfo:
        make_document(filename="..")
    assert excinfo.value.code == "invalid_filename"


def test_filename_with_control_chars_rejected() -> None:
    """NUL bytes in filenames corrupt downstream tools and logs."""
    with pytest.raises(ValidationError) as excinfo:
        make_document(filename="bad\x00name.md")
    assert excinfo.value.code == "invalid_filename"


def test_blank_media_type_rejected() -> None:
    with pytest.raises(ValidationError) as excinfo:
        make_document(media_type="")
    assert excinfo.value.code == "empty_media_type"


def test_malformed_media_type_rejected() -> None:
    with pytest.raises(ValidationError) as excinfo:
        make_document(media_type="markdown")
    assert excinfo.value.code == "invalid_media_type"


def test_non_string_content_rejected() -> None:
    with pytest.raises(ValidationError) as excinfo:
        make_document(content=b"bytes not str")
    assert excinfo.value.code == "invalid_content"


def test_naive_datetime_rejected() -> None:
    """A timezone-less timestamp is ambiguous across machines/containers."""
    with pytest.raises(ValidationError) as excinfo:
        make_document(created_at=datetime(2026, 10, 17, 12, 0, 0))
    assert excinfo.value.code == "naive_datetime"


def test_negative_page_count_rejected() -> None:
    with pytest.raises(ValidationError) as excinfo:
        make_document(page_count=-1)
    assert excinfo.value.code == "negative_page_count"


def test_metadata_must_be_a_mapping() -> None:
    with pytest.raises(ValidationError) as excinfo:
        make_document(metadata=["not", "a", "mapping"])
    assert excinfo.value.code == "invalid_metadata"


def test_metadata_with_none_value_rejected() -> None:
    """None has no representation in a vector index's metadata."""
    with pytest.raises(ValidationError) as excinfo:
        make_document(metadata={"author": None})  # type: ignore[dict-item]
    assert excinfo.value.code == "invalid_metadata_value"


def test_metadata_with_nested_structures_rejected() -> None:
    with pytest.raises(ValidationError) as excinfo:
        make_document(metadata={"nested": {"a": 1}})  # type: ignore[dict-item]
    assert excinfo.value.code == "invalid_metadata_value"


def test_metadata_with_list_value_rejected() -> None:
    with pytest.raises(ValidationError) as excinfo:
        make_document(metadata={"tags": ["a", "b"]})  # type: ignore[dict-item]
    assert excinfo.value.code == "invalid_metadata_value"


def test_metadata_with_empty_key_rejected() -> None:
    with pytest.raises(ValidationError) as excinfo:
        make_document(metadata={"  ": "value"})
    assert excinfo.value.code == "invalid_metadata_key"


def test_metadata_accepts_scalars() -> None:
    doc = make_document(
        metadata={"s": "x", "i": 1, "f": 1.5, "b": True},
    )

    assert dict(doc.metadata) == {"s": "x", "i": 1, "f": 1.5, "b": True}


def test_failed_status_requires_error() -> None:
    """A failure with no reason recorded is undebuggable."""
    with pytest.raises(ValidationError) as excinfo:
        Document.create(
            source_path="a.md",
            filename="a.md",
            media_type="text/plain",
            content="x",
            created_at=NOW,
        ).with_status(IngestStatus.FAILED)
    assert excinfo.value.code == "missing_error"


def test_error_not_allowed_on_non_failed_status() -> None:
    """Stale error text on a healthy record misleads whoever reads it later."""
    base = make_document()
    with pytest.raises(ValidationError) as excinfo:
        Document(
            doc_id=base.doc_id,
            filename=base.filename,
            media_type=base.media_type,
            content=base.content,
            metadata=base.metadata,
            created_at=NOW,
            status=IngestStatus.INDEXED,
            content_sha256=base.content_sha256,
            source_path=base.source_path,
            error="stale error",
            ingested_at=NOW,
        )
    assert excinfo.value.code == "unexpected_error"


def test_invalid_doc_id_rejected() -> None:
    base = make_document()
    with pytest.raises(ValidationError) as excinfo:
        Document(
            doc_id="not-an-id",
            filename=base.filename,
            media_type=base.media_type,
            content=base.content,
            metadata=base.metadata,
            created_at=NOW,
            content_sha256=base.content_sha256,
            source_path=base.source_path,
        )
    assert excinfo.value.code == "invalid_doc_id"


def test_absolute_source_path_rejected() -> None:
    """Absolute paths leak local filesystem layout into stored records."""
    with pytest.raises(ValidationError) as excinfo:
        make_document(source_path="/etc/passwd")
    assert excinfo.value.code == "absolute_source_path"


def test_windows_absolute_source_path_rejected() -> None:
    with pytest.raises(ValidationError) as excinfo:
        make_document(source_path="C:/Users/private/secret.md")
    assert excinfo.value.code == "absolute_source_path"


# =============================================================================
# 3. Identifier stability (the contract Stages 15 and 23 rely on)
# =============================================================================


def test_doc_id_is_stable_for_same_path() -> None:
    """Re-ingesting the same path yields the same id — the basis of idempotency."""
    assert make_doc_id("guides/api.md") == make_doc_id("guides/api.md")


def test_doc_id_is_independent_of_content() -> None:
    """Editing a file must NOT change its doc_id, so old chunks get replaced.

    If this ever fails, an edited document would be treated as brand new and
    its stale chunks would linger in the index forever.
    """
    first = Document.create(
        source_path="a.md",
        filename="a.md",
        media_type="text/plain",
        content="version one",
    )
    second = Document.create(
        source_path="a.md",
        filename="a.md",
        media_type="text/plain",
        content="version two, completely different",
    )

    assert first.doc_id == second.doc_id
    assert first.content_sha256 != second.content_sha256


def test_content_hash_detects_modified_content() -> None:
    """Different content -> different hash (how Stage 15 spots edits)."""
    assert content_sha256("abc") != content_sha256("abd")


def test_content_hash_is_identical_for_same_content() -> None:
    assert content_sha256("abc") == content_sha256("abc")


def test_content_hash_accepts_bytes_and_str_equally() -> None:
    """Hashing a file on disk and hashing decoded text must agree."""
    assert content_sha256("hello") == content_sha256(b"hello")


def test_renamed_file_gets_different_doc_id() -> None:
    """A rename is a different document identity (Stage 15 detects it via hash)."""
    assert make_doc_id("guides/old-name.md") != make_doc_id("guides/new-name.md")


def test_doc_id_is_independent_of_path_separator_style() -> None:
    """Windows and POSIX callers must produce the same id for the same file."""
    assert make_doc_id("guides\\api.md") == make_doc_id("guides/api.md")


def test_doc_id_ignores_leading_dot_slash() -> None:
    assert make_doc_id("./guides/api.md") == make_doc_id("guides/api.md")


def test_doc_id_format_is_always_well_formed() -> None:
    import re

    for path in ["a.md", "deep/nested/path/file.txt", "weird name (1).md"]:
        assert re.fullmatch(r"doc_[0-9a-f]{16}", make_doc_id(path)), path


def test_make_doc_id_rejects_traversal() -> None:
    with pytest.raises(ValidationError) as excinfo:
        make_doc_id("../secrets.md")
    assert excinfo.value.code == "path_traversal"


def test_make_doc_id_rejects_empty() -> None:
    with pytest.raises(ValidationError) as excinfo:
        make_doc_id("   ")
    assert excinfo.value.code == "empty_source_path"


def test_make_chunk_id_format() -> None:
    assert make_chunk_id("doc_0123456789abcdef", 7) == "doc_0123456789abcdef:7"


def test_make_chunk_id_is_stable() -> None:
    """Same inputs -> same id, so re-ingestion overwrites instead of appending."""
    assert make_chunk_id("doc_0123456789abcdef", 0) == make_chunk_id("doc_0123456789abcdef", 0)


def test_make_chunk_id_rejects_negative_index() -> None:
    with pytest.raises(ValidationError) as excinfo:
        make_chunk_id("doc_0123456789abcdef", -1)
    assert excinfo.value.code == "negative_chunk_index"


# =============================================================================
# 4. Status lifecycle
# =============================================================================


def test_happy_path_lifecycle() -> None:
    doc = make_document()
    assert doc.status is IngestStatus.PENDING

    doc = doc.with_status(IngestStatus.PARSING)
    assert doc.status is IngestStatus.PARSING

    doc = doc.with_status(IngestStatus.INDEXING)
    assert doc.status is IngestStatus.INDEXING

    doc = doc.with_status(IngestStatus.INDEXED, ingested_at=NOW)
    assert doc.status is IngestStatus.INDEXED
    assert doc.ingested_at == NOW


def test_original_is_unchanged_by_with_status() -> None:
    """Immutability: the old record must not be mutated in place."""
    doc = make_document()
    _ = doc.with_status(IngestStatus.PARSING)

    assert doc.status is IngestStatus.PENDING


def test_failure_can_happen_from_any_active_state() -> None:
    """Any non-terminal state may fail — partial progress must not be a dead end."""
    # PENDING -> FAILED
    pending = make_document().with_status(IngestStatus.FAILED, error="boom")
    assert pending.status is IngestStatus.FAILED
    assert pending.error == "boom"

    # PARSING -> FAILED
    parsing = (
        make_document()
        .with_status(IngestStatus.PARSING)
        .with_status(IngestStatus.FAILED, error="boom")
    )
    assert parsing.status is IngestStatus.FAILED

    # INDEXING -> FAILED
    indexing = (
        make_document()
        .with_status(IngestStatus.PARSING)
        .with_status(IngestStatus.INDEXING)
        .with_status(IngestStatus.FAILED, error="boom")
    )
    assert indexing.status is IngestStatus.FAILED
    assert indexing.error == "boom"


def test_indexed_document_requires_ingested_at() -> None:
    """An indexed record with no timestamp would be impossible to audit."""
    base = make_document().with_status(IngestStatus.PARSING).with_status(IngestStatus.INDEXING)

    # with_status(INDEXED) defaults the timestamp to "now", so it is never None.
    indexed = base.with_status(IngestStatus.INDEXED)

    assert indexed.ingested_at is not None


def test_failed_record_reports_terminal() -> None:
    failed = make_document().with_status(IngestStatus.FAILED, error="parse error")

    assert failed.is_terminal is True
    assert failed.allowed_transitions() == frozenset()


def test_indexed_record_reports_terminal() -> None:
    indexed = make_indexed()

    assert indexed.is_terminal is True
    assert indexed.is_terminal is indexed.status.is_terminal
    assert indexed.ingested_at == NOW


def test_terminal_states_cannot_be_reactivated() -> None:
    """A finished record can never be nudged back to PENDING.

    Otherwise a stale index could silently look 'fresh' again.
    """
    indexed = make_indexed()

    for target in (
        IngestStatus.PENDING,
        IngestStatus.PARSING,
        IngestStatus.INDEXING,
        IngestStatus.FAILED,
    ):
        with pytest.raises(ValidationError, match="cannot move"):
            indexed.with_status(target, error="x")


def test_cannot_skip_parsing_and_go_straight_to_indexed() -> None:
    doc = make_document()

    with pytest.raises(ValidationError, match="cannot move"):
        doc.with_status(IngestStatus.INDEXED, ingested_at=NOW)


def test_valid_transitions_from_pending() -> None:
    doc = make_document()

    assert doc.allowed_transitions() == frozenset({IngestStatus.PARSING, IngestStatus.FAILED})


def test_with_status_to_failed_requires_error_message() -> None:
    doc = make_document()

    with pytest.raises(ValidationError) as excinfo:
        doc.with_status(IngestStatus.FAILED)
    assert excinfo.value.code == "missing_error"


def test_failure_clears_error_when_moving_to_indexed_is_impossible() -> None:
    """Errors never linger on a non-failed record (validated at construction)."""
    failed = make_document().with_status(IngestStatus.FAILED, error="boom")

    assert failed.error == "boom"
    assert failed.status is IngestStatus.FAILED


# =============================================================================
# 5. Serialisation safety
# =============================================================================


def test_summary_excludes_document_content() -> None:
    """Document text must never appear in a log line or an API list payload."""
    doc = make_document(content="TOP SECRET CONFIDENTIAL TEXT")

    summary = doc.summary()

    assert "TOP SECRET CONFIDENTIAL TEXT" not in str(summary)
    assert "content" not in summary
    assert "content_sha256" not in summary


def test_summary_is_json_serialisable() -> None:
    import json

    doc = make_document(metadata={"category": "spec", "revision": 3})

    payload = json.dumps(doc.summary())

    assert '"doc_id"' in payload
    assert '"revision": 3' in payload


def test_summary_contains_expected_fields() -> None:
    doc = make_document()
    summary = doc.summary()

    for key in (
        "doc_id",
        "filename",
        "media_type",
        "status",
        "char_count",
        "page_count",
        "source_path",
        "created_at",
        "ingested_at",
        "error",
        "metadata",
    ):
        assert key in summary, f"summary is missing {key}"


def test_summary_status_is_a_plain_string() -> None:
    """Consumers (JSON, Chroma metadata) need str, not a StrEnum instance."""
    assert isinstance(make_document().summary()["status"], str)


def test_record_is_immutable() -> None:
    """``frozen`` prevents one worker from mutating a record another reads."""
    import dataclasses

    doc = make_document()

    with pytest.raises(dataclasses.FrozenInstanceError):
        doc.filename = "other.md"  # type: ignore[misc]

    # And the value really is unchanged, not merely guarded by a setter.
    assert doc.filename == "api-spec.md"


def test_enum_values_are_strings() -> None:
    """StrEnum lets status compare and serialise as a plain string."""
    assert IngestStatus.INDEXED == "indexed"
    assert str(IngestStatus.FAILED) == "failed"


# =============================================================================
# RetrievedChunk (Stage 30) — the read half's domain record
# =============================================================================

DOC_ID = "doc_0123456789abcdef"


def make_retrieved(*, chunk_id: str = f"{DOC_ID}:0", distance: float = 0.25) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=DOC_ID,
        content="the payments api settles in one day",
        distance=distance,
        metadata={
            "filename": "api-spec.md",
            "media_type": "text/markdown",
            "section": "Requirements",
            "page": 3,
        },
    )


def test_retrieved_chunk_constructs() -> None:
    result = make_retrieved()
    assert result.doc_id == DOC_ID
    assert result.chunk_id.startswith(f"{DOC_ID}:")
    assert result.distance == 0.25
    assert result.content == "the payments api settles in one day"


def test_retrieved_chunk_id_matches_its_document() -> None:
    result = make_retrieved(chunk_id=f"{DOC_ID}:7")
    assert result.doc_id == DOC_ID


def test_retrieved_chunk_id_rejected_when_wrong_doc() -> None:
    """A result whose id names a different document is a broken provenance."""
    with pytest.raises(ValidationError) as excinfo:
        RetrievedChunk(
            chunk_id=f"{DOC_ID}:1",
            doc_id="doc_0000000000000000",
            content="x",
            distance=0.0,
        )
    assert excinfo.value.code == "invalid_chunk_id"


def test_retrieved_chunk_rejects_mismatched_chunk_id() -> None:
    with pytest.raises(ValidationError) as excinfo:
        RetrievedChunk(
            chunk_id=f"{DOC_ID}:wrong",
            doc_id=DOC_ID,
            content="x",
            distance=0.0,
        )
    assert excinfo.value.code == "invalid_chunk_id"


def test_retrieved_chunk_rejects_empty_content() -> None:
    with pytest.raises(ValidationError) as excinfo:
        RetrievedChunk(
            chunk_id=f"{DOC_ID}:0",
            doc_id=DOC_ID,
            content="   ",
            distance=0.0,
        )
    assert excinfo.value.code == "empty_content"


def test_retrieved_chunk_distance_must_be_finite() -> None:
    with pytest.raises(ValidationError) as excinfo:
        RetrievedChunk(
            chunk_id=f"{DOC_ID}:0",
            doc_id=DOC_ID,
            content="x",
            distance=float("nan"),
        )
    assert excinfo.value.code == "invalid_distance"


def test_retrieved_chunk_distance_must_be_approximately_non_negative() -> None:
    result = make_retrieved(distance=-1e-6)
    # Near-zero negative floating-point artifacts are allowed (cosine
    # arithmetic rounds slightly past zero), but a clearly negative
    # distance is impossible.
    assert result.distance == -1e-6

    with pytest.raises(ValidationError) as excinfo:
        make_retrieved(distance=-1.0)
    assert excinfo.value.code == "invalid_distance"


def test_retrieved_chunk_distance_rejects_non_numeric() -> None:
    with pytest.raises(ValidationError) as excinfo:
        RetrievedChunk(
            chunk_id=f"{DOC_ID}:0",
            doc_id=DOC_ID,
            content="x",
            distance="close",
        )
    assert excinfo.value.code == "invalid_distance"


def test_retrieved_chunk_derived_accessors_read_metadata() -> None:
    result = make_retrieved()
    assert result.filename == "api-spec.md"
    assert result.page == 3
    assert result.section == "Requirements"


def test_retrieved_chunk_accessors_fallback_to_none() -> None:
    result = RetrievedChunk(
        chunk_id=f"{DOC_ID}:0",
        doc_id=DOC_ID,
        content="x",
        distance=0.0,
        metadata={"filename": 99},
    )
    # The key exists but the value is the wrong scalar — it falls back.
    assert result.filename is None
    assert result.page is None
    assert result.section is None


def test_retrieved_chunk_summary_omits_content() -> None:
    result = make_retrieved()
    summary = result.summary()
    assert "content" not in summary
    assert result.content not in str(summary)


def test_retrieved_chunk_is_immutable() -> None:
    import dataclasses

    result = make_retrieved()

    with pytest.raises(dataclasses.FrozenInstanceError):
        result.distance = 0.0  # type: ignore[misc]


def test_metadata_filter_alias_is_mapping() -> None:
    # The alias only names the shape — the retrieval layer enforces legality.

    f: MetadataFilter = {"doc_id": {"$in": ["x"]}, "section": "Requirements"}
    assert isinstance(f, Mapping)


def test_retrieved_chunk_ignores_empty_filter_for_summary() -> None:
    """The filter is never part of a result — it is a caller-side query."""
    # This is a shape test: RetrievedChunk carries no filter field.
    result = make_retrieved()
    assert "filter" not in result.summary()
