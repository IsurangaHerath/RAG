"""Tests for the document registry (``app/storage/registry.py``).

Real SQLite against a temporary file on every test — the connection is
closed on the way out so Windows can delete the directory again. What
gets pinned here: exact round-trips through the model's own validation,
``create`` as the fresh record that restarts a terminal document, the
status graph enforced *through* ``Document.with_status`` rather than
beside it, deterministic ``find_by_hash`` when one hash backs several
paths, and sqlite's failures arriving as ``DependencyError``. The last
test seeds an ``IngestPipeline`` from a real registry, closing the loop
section 5.16 left open.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.domain.errors import DependencyError, NotFoundError, ValidationError
from app.domain.models import Document, IngestStatus, content_sha256
from app.ingestion.pipeline import IngestPipeline, PipelineAction
from app.storage.registry import DocumentRegistry
from tests.fakes import FakeChunkStore, FakeEmbedder

SOURCE_A = "guides/api-spec.md"
SOURCE_B = "guides/api-copy.md"
TEXT_A = "# Limits\n\nBursts are smoothed per account.\n"
TEXT_B = "# Limits\n\nBursts are smoothed per account, revised.\n"


def make_document(
    source_path: str = SOURCE_A,
    content: str = TEXT_A,
    **overrides: object,
) -> Document:
    """A registered-shaped document with the usual metadata."""
    return Document.create(
        source_path=source_path,
        filename=Path(source_path).name,
        media_type="text/markdown",
        content=content,
        metadata={
            "filename": Path(source_path).name,
            "media_type": "text/markdown",
            "starred": True,
            "revision": 3,
            "section": "Límits & quotes",
        },
        **overrides,  # type: ignore[arg-type]
    )


@pytest.fixture()
def registry(tmp_path: Path) -> DocumentRegistry:
    """A fresh registry per test, always closed on the way out."""
    instance = DocumentRegistry(path=tmp_path / "registry.sqlite3")
    yield instance
    instance.close()


# --- Round-trips ------------------------------------------------------------


def test_create_then_get_returns_the_same_document(
    registry: DocumentRegistry,
) -> None:
    """Full fidelity: text, checksum, status, tz-aware dates, metadata."""
    doc = make_document()

    registry.create(doc)
    loaded = registry.get(doc.doc_id)

    assert loaded == doc
    assert loaded is not None
    assert loaded.created_at.tzinfo is not None
    assert loaded.metadata["starred"] is True
    assert loaded.metadata["section"] == "Límits & quotes"


def test_getting_an_unknown_document_returns_none(
    registry: DocumentRegistry,
) -> None:
    """Absence is ``None``, not an exception — section 5.3's signature says so."""
    assert registry.get("doc_0000000000000000") is None


def test_registering_the_same_id_replaces_every_column(
    registry: DocumentRegistry,
) -> None:
    """``create`` is the upsert: new checksum and text win, one row stays."""
    original = make_document(content=TEXT_A)
    registry.create(original)

    edited = make_document(content=TEXT_B)
    registry.create(edited)

    rows = registry.list()
    assert len(rows) == 1
    assert rows[0] == edited
    assert rows[0].content_sha256 == content_sha256(TEXT_B)


def test_a_terminal_record_can_be_restarted_by_a_fresh_registration(
    registry: DocumentRegistry,
) -> None:
    """section 5.1's remedy for terminal states: re-ingest as a fresh record."""
    doc = make_document()
    registry.create(doc)
    registry.update_status(doc.doc_id, IngestStatus.PARSING)
    registry.update_status(doc.doc_id, IngestStatus.INDEXING)
    registry.update_status(doc.doc_id, IngestStatus.INDEXED)

    registry.create(make_document(content=TEXT_B))

    restarted = registry.get(doc.doc_id)
    assert restarted is not None
    assert restarted.status is IngestStatus.PENDING
    assert restarted.content_sha256 == content_sha256(TEXT_B)
    assert restarted.ingested_at is None


def test_data_survives_a_new_instance_on_the_same_file(tmp_path: Path) -> None:
    """Persistence is the whole point of a registry over an in-memory list."""
    path = tmp_path / "registry.sqlite3"
    first = DocumentRegistry(path=path)
    first.create(make_document())
    first.close()

    second = DocumentRegistry(path=path)
    try:
        assert second.get(make_document().doc_id) is not None
        assert second.path == path
    finally:
        second.close()


# --- Listing ----------------------------------------------------------------


def test_list_returns_every_record_in_stable_order(
    registry: DocumentRegistry,
) -> None:
    """Ordered by ``doc_id``: paging never skips or repeats a row."""
    for path in ("guides/c.md", "guides/a.md", "guides/b.md"):
        registry.create(make_document(source_path=path))

    listed = registry.list()

    assert [doc.source_path for doc in listed] == [
        doc.source_path for doc in sorted(listed, key=lambda d: d.doc_id)
    ]
    assert len(listed) == 3


def test_list_filters_by_status(registry: DocumentRegistry) -> None:
    """The CLI's index report and the API's list both lean on this."""
    done = make_document(source_path="guides/done.md")
    waiting = make_document(source_path="guides/waiting.md")
    for doc in (done, waiting):
        registry.create(doc)
    registry.update_status(done.doc_id, IngestStatus.PARSING)
    registry.update_status(done.doc_id, IngestStatus.INDEXING)
    registry.update_status(done.doc_id, IngestStatus.INDEXED)

    indexed = registry.list(status=IngestStatus.INDEXED)
    pending = registry.list(status=IngestStatus.PENDING)

    assert [doc.doc_id for doc in indexed] == [done.doc_id]
    assert [doc.doc_id for doc in pending] == [waiting.doc_id]


def test_list_pages_with_limit_and_offset(registry: DocumentRegistry) -> None:
    """Two arguments, three pages' worth of behaviour."""
    for path in ("guides/a.md", "guides/b.md", "guides/c.md"):
        registry.create(make_document(source_path=path))
    all_ids = [doc.doc_id for doc in registry.list()]

    page_one = [doc.doc_id for doc in registry.list(limit=2)]
    page_two = [doc.doc_id for doc in registry.list(limit=2, offset=2)]

    assert page_one == all_ids[:2]
    assert page_two == all_ids[2:]


def test_list_of_an_empty_registry_is_empty(registry: DocumentRegistry) -> None:
    """No rows, no surprise: an empty list, not an error."""
    assert registry.list() == []
    assert registry.list(status=IngestStatus.FAILED) == []


def test_negative_paging_is_rejected(registry: DocumentRegistry) -> None:
    """sqlite reads ``LIMIT -1`` as unbounded; a paging call must not dump."""
    with pytest.raises(ValidationError) as excinfo:
        registry.list(limit=-1)
    assert excinfo.value.code == "invalid_request"

    with pytest.raises(ValidationError):
        registry.list(offset=-1)


# --- The status lifecycle ---------------------------------------------------


def test_status_walks_the_documented_path(registry: DocumentRegistry) -> None:
    """pending → parsing → indexing → indexed, with the stamp at the end."""
    doc = make_document()
    registry.create(doc)

    registry.update_status(doc.doc_id, IngestStatus.PARSING)
    registry.update_status(doc.doc_id, IngestStatus.INDEXING)
    assert registry.get(doc.doc_id) is not None
    assert registry.get(doc.doc_id).ingested_at is None  # type: ignore[union-attr]

    registry.update_status(doc.doc_id, IngestStatus.INDEXED)

    finished = registry.get(doc.doc_id)
    assert finished is not None
    assert finished.status is IngestStatus.INDEXED
    assert finished.ingested_at is not None
    assert finished.error is None


def test_a_terminal_record_refuses_further_moves(registry: DocumentRegistry) -> None:
    """Finished means finished: no nudging a stale index back to life."""
    doc = make_document()
    registry.create(doc)
    registry.update_status(doc.doc_id, IngestStatus.FAILED, error="PARSE_ERROR: broken")

    with pytest.raises(ValidationError) as excinfo:
        registry.update_status(doc.doc_id, IngestStatus.PARSING)
    assert excinfo.value.code == "invalid_status_transition"


def test_failure_is_reachable_and_keeps_its_error(
    registry: DocumentRegistry,
) -> None:
    """Any working state may fail; the reason survives the round-trip."""
    doc = make_document()
    registry.create(doc)
    registry.update_status(doc.doc_id, IngestStatus.FAILED, error="PARSE_ERROR: encrypted pdf")

    stored = registry.get(doc.doc_id)

    assert stored is not None
    assert stored.status is IngestStatus.FAILED
    assert stored.error == "PARSE_ERROR: encrypted pdf"


def test_failure_without_an_error_is_rejected(registry: DocumentRegistry) -> None:
    """The model's ``missing_error`` rule, enforced through the registry."""
    doc = make_document()
    registry.create(doc)

    with pytest.raises(ValidationError) as excinfo:
        registry.update_status(doc.doc_id, IngestStatus.FAILED, error=None)
    assert excinfo.value.code == "missing_error"


def test_an_illegal_jump_is_rejected(registry: DocumentRegistry) -> None:
    """pending → indexed skips the whole graph; the model says no."""
    doc = make_document()
    registry.create(doc)

    with pytest.raises(ValidationError) as excinfo:
        registry.update_status(doc.doc_id, IngestStatus.INDEXED)
    assert excinfo.value.code == "invalid_status_transition"


def test_updating_a_document_that_was_never_registered(
    registry: DocumentRegistry,
) -> None:
    """Status changes cannot queue behind a record that does not exist."""
    with pytest.raises(NotFoundError):
        registry.update_status("doc_0000000000000000", IngestStatus.PARSING)


# --- Dedup counterpart ------------------------------------------------------


def test_find_by_hash_returns_the_lowest_doc_id(
    registry: DocumentRegistry,
) -> None:
    """One hash can back several paths; 'first' must be the same every time."""
    first = make_document(source_path="guides/a-copy.md", content=TEXT_A)
    second = make_document(source_path="guides/z-copy.md", content=TEXT_A)
    registry.create(second)
    registry.create(first)

    found = registry.find_by_hash(content_sha256(TEXT_A))

    assert found is not None
    assert found.doc_id == min(first.doc_id, second.doc_id)


def test_find_by_hash_miss_returns_none(registry: DocumentRegistry) -> None:
    """Same contract as dedup's sequence scan — absence is ``None``."""
    registry.create(make_document())
    assert registry.find_by_hash("0" * 64) is None


# --- Failure translation ----------------------------------------------------


def test_an_unopenable_path_becomes_a_dependency_error(tmp_path: Path) -> None:
    """sqlite cannot open a directory as a database; callers see 502-shape."""
    with pytest.raises(DependencyError) as excinfo:
        DocumentRegistry(path=tmp_path)
    assert excinfo.value.code == "dependency_unavailable"


# --- The cross-stage proof --------------------------------------------------


class CountingEmbedder:
    """FakeEmbedder that counts calls, to prove a skip costs nothing."""

    def __init__(self) -> None:
        self._fake = FakeEmbedder()
        self.calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return self._fake.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        self.calls += 1
        return self._fake.embed_query(text)


def test_a_pipeline_seeded_from_the_registry_skips_unchanged_files(
    tmp_path: Path,
) -> None:
    """section 5.16's stand-in list replaced by the real thing: no embed, no write."""
    source = tmp_path / "api-spec.md"
    source.write_text(TEXT_A, encoding="utf-8")
    doc = make_document(content=TEXT_A)
    store_reg = DocumentRegistry(path=tmp_path / "registry.sqlite3")
    store_reg.create(doc)
    known = store_reg.list()
    store_reg.close()

    store = FakeChunkStore()
    embedder = CountingEmbedder()
    pipeline = IngestPipeline(embedder=embedder, store=store, known=known)  # type: ignore[arg-type]

    result = pipeline.run(source, source_path=SOURCE_A)

    assert result.action is PipelineAction.SKIPPED
    assert result.code == "unchanged"
    assert embedder.calls == 0
    assert store.ops == []
