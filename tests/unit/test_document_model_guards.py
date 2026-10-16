"""Tests for guards only reachable via direct ``Document(...)`` construction.

``Document.create()`` is the recommended entry point, but the dataclass is
public and several callers (deserialisation, tests, the storage layer
rehydrating a stored record) will build instances field-by-field. These
tests make sure validation still holds on that path — otherwise bypassing
``create()`` would bypass every rule.

Companion to ``test_document_model.py``.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from app.domain.errors import ValidationError
from app.domain.models import Document, IngestStatus, make_doc_id


def bare(**overrides: object) -> dict[str, object]:
    """Minimal valid kwargs for direct construction."""
    params: dict[str, object] = {
        "doc_id": make_doc_id("a.md"),
        "filename": "a.md",
        "media_type": "text/plain",
        "content": "hello",
        "metadata": {},
        "created_at": datetime(2026, 10, 17, tzinfo=__import__("datetime").UTC),
        "content_sha256": "0" * 64,
        "source_path": "a.md",
    }
    params.update(overrides)
    return params


def test_direct_construction_valid_record() -> None:
    """The happy path for field-by-field construction."""
    doc = Document(**bare())  # type: ignore[arg-type]

    assert doc.filename == "a.md"


def test_created_at_must_be_a_datetime() -> None:
    with pytest.raises(ValidationError) as excinfo:
        Document(**bare(created_at="2026-10-17T00:00:00Z"))  # type: ignore[arg-type]
    assert excinfo.value.code == "invalid_created_at"


def test_naive_ingested_at_rejected() -> None:
    """Inconsistent with created_at: both timestamps must be timezone-aware."""
    with pytest.raises(ValidationError) as excinfo:
        Document(**bare(ingested_at=datetime(2026, 10, 17)))  # type: ignore[arg-type]
    assert excinfo.value.code == "naive_datetime"


def test_status_must_be_an_ingest_status() -> None:
    with pytest.raises(ValidationError) as excinfo:
        Document(**bare(status="indexed"))  # type: ignore[arg-type]
    assert excinfo.value.code == "invalid_status"


def test_failed_status_without_error_rejected_on_construction() -> None:
    """The cross-field rule applies to create(), with_status() AND direct init."""
    with pytest.raises(ValidationError) as excinfo:
        Document(**bare(status=IngestStatus.FAILED))  # type: ignore[arg-type]
    assert excinfo.value.code == "missing_error"


def test_absolute_source_path_rejected_on_construction() -> None:
    """create() derives doc_id first, so this guard only fires on direct init."""
    with pytest.raises(ValidationError) as excinfo:
        Document(**bare(source_path="/etc/shadow"))
    assert excinfo.value.code == "absolute_source_path"


def test_indexed_without_ingested_at_rejected() -> None:
    """An indexed record with no timestamp could never be audited."""
    with pytest.raises(ValidationError) as excinfo:
        Document(**bare(status=IngestStatus.INDEXED))  # type: ignore[arg-type]
    assert excinfo.value.code == "missing_ingested_at"


def test_non_mapping_metadata_rejected_on_construction() -> None:
    """create() validates this before coercion; direct init must too."""
    with pytest.raises(ValidationError) as excinfo:
        Document(**bare(metadata=["a", "b"]))  # type: ignore[arg-type]
    assert excinfo.value.code == "invalid_metadata"


def test_doc_id_with_wrong_length_rejected() -> None:
    with pytest.raises(ValidationError) as excinfo:
        Document(**bare(doc_id="doc_abc"))
    assert excinfo.value.code == "invalid_doc_id"


def test_doc_id_without_prefix_rejected() -> None:
    with pytest.raises(ValidationError) as excinfo:
        Document(**bare(doc_id="0123456789abcdef"))
    assert excinfo.value.code == "invalid_doc_id"


# --- make_doc_id edge case ----------------------------------------------------


def test_doc_id_from_path_that_resolves_to_empty() -> None:
    """'./' normalises to an empty path, which must not silently hash to a valid id."""
    with pytest.raises(ValidationError) as excinfo:
        make_doc_id("./")
    assert excinfo.value.code == "empty_source_path"


def test_doc_id_from_dot_only_path() -> None:
    """A bare '.' is a directory reference, not a document path."""
    # '.' has no traversal segment and no separator, so it normalises to '.'
    # and is accepted — but it must still produce a well-formed id.
    result = make_doc_id(".")
    assert result.startswith("doc_")
    assert len(result) == 20


# --- sanitisation on create ----------------------------------------------------


def test_backslash_source_path_normalised_to_forward_slash() -> None:
    """A Windows-authored path is stored in canonical form."""
    doc = Document(**bare(source_path="guides\\api.md"))  # type: ignore[arg-type]

    assert doc.source_path == "guides/api.md"
