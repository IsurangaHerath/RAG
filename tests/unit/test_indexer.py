"""Tests for the batch indexing pass (``app/ingestion/indexer.py``).

The module exists to make a whole directory *registered, written and
status-tracked* in one pass, so most of these tests read three things back
afterwards: the store's contents, the registry row, and the outcome — and
assert the three agree.

The ordering rules get the most attention, because that is what the module
claims: a row exists before any chunk does, ``INDEXED`` is only ever
recorded after the write returns, and a document whose previous attempt
failed is retried rather than mistaken for something unchanged.

Everything is offline: files in ``tmp_path``, an in-memory store, the
offline embedder double — plus one test against a real on-disk Chroma
index to prove the structural fit end to end.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from app.domain.errors import DependencyError, NotFoundError
from app.domain.models import Document, IngestStatus, make_doc_id
from app.embeddings.base import Embedder
from app.ingestion.deduplication import ChangeKind
from app.ingestion.indexer import (
    IndexOutcome,
    IndexResult,
    _all_records,
    _error_line,
    index_directory,
)
from app.ingestion.pipeline import (
    NO_CHUNKS_CODE,
    RENAMED_CODE,
    UNCHANGED_CODE,
    ChunkSplitter,
    ChunkStore,
)
from app.ingestion.service import EMPTY_FILE_CODE
from app.storage.chroma import ChromaIndex
from app.storage.registry import DocumentRegistry
from tests.fakes import FakeChunkStore, FakeEmbedder

TEXT = "# Payments API\n\nThe platform exposes a payments API for refunds and disputes.\n"
OTHER_TEXT = "# Rate limits\n\nBursty clients receive a 429 with a Retry-After header.\n"
MARKER = "INDEXER-CANARY-6B41"
CANARY = f"{MARKER} must never surface in a summary."
#: Invalid UTF-8 with no NUL byte: ``binary_content`` would need a NUL,
#: and a BOM prefix would make the decoder try harder — this fails as
#: ``encoding_error`` for certain.
BROKEN_BYTES = b"\xff\xfa\xca\xfe this is not utf-8 text \xe5\xe8"


class FailingEmbedder:
    """An embedder that refuses every batch — the provider is down."""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        raise DependencyError(
            "The embedding provider refused the batch.", code="dependency_unavailable"
        )

    def embed_query(self, text: str) -> list[float]:
        raise DependencyError(
            "The embedding provider refused the query.", code="dependency_unavailable"
        )


class StatusCheckingEmbedder:
    """Record the registry status visible at the moment embedding begins."""

    def __init__(self, registry: DocumentRegistry, doc_id: str) -> None:
        self._registry = registry
        self._doc_id = doc_id
        self._fake = FakeEmbedder()
        self.seen: IngestStatus | None = None

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        record = self._registry.get(self._doc_id)
        self.seen = record.status if record is not None else None
        return self._fake.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._fake.embed_query(text)


class EmptyChunker:
    """A chunker whose documents never yield anything (the no_chunks path)."""

    def split(self, document: object, *, doc_id: str) -> list[object]:
        return []


def write(root: Path, name: str, text: str) -> Path:
    """Create a corpus file under the root, making parent folders."""
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def run_pass(
    root: Path,
    registry: DocumentRegistry,
    *,
    embedder: Embedder | None = None,
    store: ChunkStore | None = None,
    chunker: ChunkSplitter | None = None,
    recursive: bool = True,
) -> IndexResult:
    """One indexing pass with fresh defaults for whatever is not given."""
    return index_directory(
        root,
        embedder=embedder if embedder is not None else FakeEmbedder(),
        store=store if store is not None else FakeChunkStore(),
        registry=registry,
        chunker=chunker,
        recursive=recursive,
    )


def by_path(result: IndexResult) -> dict[str, IndexOutcome]:
    """Outcomes keyed by relative path, for order-independent assertions."""
    return {outcome.file.relative_path: outcome for outcome in result.outcomes}


@pytest.fixture
def registry(tmp_path: Path) -> DocumentRegistry:
    """A fresh registry on its own temporary file."""
    return DocumentRegistry(path=tmp_path / "registry.sqlite3")


# --- Creating: what a first pass must produce --------------------------------


def test_a_new_file_is_written_registered_and_marked_indexed(
    tmp_path: Path, registry: DocumentRegistry
) -> None:
    root = tmp_path / "corpus"
    write(root, "guide.md", TEXT)
    store = FakeChunkStore()

    result = run_pass(root, registry, store=store)
    outcome = by_path(result)["guide.md"]

    assert outcome.is_created
    assert outcome.written
    assert outcome.chunk_count > 0
    assert store.count(outcome.doc_id) == outcome.chunk_count
    assert result.created_count == 1
    assert result.chunks_written == outcome.chunk_count
    assert not result.has_failures

    row = registry.get(outcome.doc_id)
    assert row is not None
    assert row.status is IngestStatus.INDEXED
    assert row.ingested_at is not None
    assert (
        row.content_sha256
        == Document.create(
            source_path="guide.md",
            filename="guide.md",
            media_type="text/markdown",
            content=TEXT,
        ).content_sha256
    )

    assert [decision.kind for decision in result.decisions] == [ChangeKind.NEW]


def test_the_status_walk_records_parsing_and_indexing_before_indexed(
    tmp_path: Path, registry: DocumentRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "corpus"
    write(root, "guide.md", TEXT)

    seen: list[IngestStatus] = []
    original = registry.update_status

    def spy(doc_id: str, status: IngestStatus, error: str | None = None) -> None:
        seen.append(status)
        original(doc_id, status, error)

    monkeypatch.setattr(registry, "update_status", spy)

    run_pass(root, registry)

    assert seen == [IngestStatus.PARSING, IngestStatus.INDEXING, IngestStatus.INDEXED]


def test_the_row_is_walked_to_indexing_before_the_embedder_runs(
    tmp_path: Path, registry: DocumentRegistry
) -> None:
    """The row-before-work claim, observed from inside the call."""
    root = tmp_path / "corpus"
    write(root, "guide.md", TEXT)
    embedder = StatusCheckingEmbedder(registry, make_doc_id("guide.md"))

    result = run_pass(root, registry, embedder=embedder)

    assert embedder.seen is IngestStatus.INDEXING
    assert result.created_count == 1


def test_unsupported_files_are_carried_through_and_not_counted_as_failures(
    tmp_path: Path, registry: DocumentRegistry
) -> None:
    root = tmp_path / "corpus"
    write(root, "guide.md", TEXT)
    (root / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    result = run_pass(root, registry)

    assert [unsupported.relative_path for unsupported in result.unsupported] == ["logo.png"]
    assert len(result.outcomes) == 1
    assert result.total_count == 2
    assert not result.has_failures


# --- Re-running: what the second pass must not do ----------------------------


def test_a_second_run_skips_the_unchanged_file_with_no_store_work(
    tmp_path: Path, registry: DocumentRegistry
) -> None:
    root = tmp_path / "corpus"
    write(root, "guide.md", TEXT)
    store = FakeChunkStore()

    run_pass(root, registry, store=store)
    second = run_pass(root, registry, store=store)
    outcome = by_path(second)["guide.md"]

    assert outcome.is_skipped
    assert outcome.code == UNCHANGED_CODE
    assert outcome.status is IngestStatus.INDEXED
    assert second.skipped_count == 1
    # Exactly the first pass's delete-then-upsert, nothing after it.
    assert store.ops == [("delete", outcome.doc_id), ("upsert", outcome.doc_id)]


def test_editing_a_file_replaces_its_chunks_and_keeps_the_row_indexed(
    tmp_path: Path, registry: DocumentRegistry
) -> None:
    root = tmp_path / "corpus"
    write(root, "guide.md", TEXT)
    store = FakeChunkStore()

    run_pass(root, registry, store=store)
    write(root, "guide.md", OTHER_TEXT)
    second = run_pass(root, registry, store=store)
    outcome = by_path(second)["guide.md"]

    assert outcome.is_replaced
    assert outcome.decision is not None
    assert outcome.decision.kind is ChangeKind.MODIFIED
    assert second.replaced_count == 1

    doc_id = make_doc_id("guide.md")
    assert store.count(doc_id) == outcome.chunk_count
    assert store.ops.count(("delete", doc_id)) == 2

    row = registry.get(doc_id)
    assert row is not None
    assert row.status is IngestStatus.INDEXED
    assert (
        row.content_sha256
        != Document.create(
            source_path="guide.md",
            filename="guide.md",
            media_type="text/markdown",
            content=TEXT,
        ).content_sha256
    )


def test_identical_content_at_a_second_path_is_skipped_as_renamed(
    tmp_path: Path, registry: DocumentRegistry
) -> None:
    root = tmp_path / "corpus"
    write(root, "a.md", TEXT)
    write(root, "copy.md", TEXT)
    store = FakeChunkStore()

    result = run_pass(root, registry, store=store)

    created = [outcome for outcome in result.outcomes if outcome.is_created]
    renamed = [outcome for outcome in result.outcomes if outcome.code == RENAMED_CODE]
    assert len(created) == 1
    assert len(renamed) == 1
    assert renamed[0].is_skipped
    # The copy has no row and no chunks: same bytes must not be indexed twice.
    assert registry.get(renamed[0].doc_id) is None
    assert store.count() == created[0].chunk_count
    assert len(registry.list(limit=10)) == 1


def test_a_failed_document_is_retried_instead_of_skipped_unchanged(
    tmp_path: Path, registry: DocumentRegistry
) -> None:
    """The INDEXED-only seeding rule: a failure row is not a known document."""
    root = tmp_path / "corpus"
    write(root, "guide.md", TEXT)
    failed = Document.create(
        source_path="guide.md",
        filename="guide.md",
        media_type="text/markdown",
        content=TEXT,
    ).with_status(IngestStatus.FAILED, error="dependency_unavailable: earlier attempt")
    registry.create(failed)
    store = FakeChunkStore()

    result = run_pass(root, registry, store=store)
    outcome = by_path(result)["guide.md"]

    assert outcome.is_created
    assert outcome.chunk_count > 0
    assert store.count(outcome.doc_id) == outcome.chunk_count
    row = registry.get(outcome.doc_id)
    assert row is not None
    assert row.status is IngestStatus.INDEXED


# --- Failures: what must be left behind --------------------------------------


def test_an_embedder_outage_leaves_a_failed_row_and_no_chunks(
    tmp_path: Path, registry: DocumentRegistry
) -> None:
    root = tmp_path / "corpus"
    write(root, "guide.md", TEXT)
    store = FakeChunkStore()

    result = run_pass(root, registry, store=store, embedder=FailingEmbedder())
    outcome = by_path(result)["guide.md"]

    assert outcome.is_failed
    assert outcome.code == "dependency_unavailable"
    assert result.has_failures
    assert store.count() == 0

    row = registry.get(outcome.doc_id)
    assert row is not None
    assert row.status is IngestStatus.FAILED
    assert "dependency_unavailable" in (row.error or "")


def test_an_unreadable_file_fails_alone_and_the_batch_continues(
    tmp_path: Path, registry: DocumentRegistry
) -> None:
    root = tmp_path / "corpus"
    write(root, "a.md", TEXT)
    (root / "b.md").write_bytes(BROKEN_BYTES)
    write(root, "c.md", OTHER_TEXT)

    result = run_pass(root, registry)
    outcomes = by_path(result)

    assert outcomes["a.md"].is_created
    assert outcomes["c.md"].is_created
    assert outcomes["b.md"].is_failed
    assert outcomes["b.md"].code == "encoding_error"
    assert result.created_count == 2
    assert result.failed_count == 1
    assert result.has_failures
    # The unreadable file never became a document.
    assert registry.get(make_doc_id("b.md")) is None


def test_a_load_failure_retires_a_stalled_pending_row(
    tmp_path: Path, registry: DocumentRegistry
) -> None:
    root = tmp_path / "corpus"
    stalled = Document.create(
        source_path="guide.md",
        filename="guide.md",
        media_type="text/markdown",
        content=TEXT,
    )
    registry.create(stalled)
    write(root, "guide.md", TEXT)
    (root / "guide.md").write_bytes(BROKEN_BYTES)

    result = run_pass(root, registry)
    outcome = by_path(result)["guide.md"]

    assert outcome.is_failed
    assert outcome.code == "encoding_error"
    assert outcome.status is IngestStatus.FAILED

    row = registry.get(outcome.doc_id)
    assert row is not None
    assert row.status is IngestStatus.FAILED
    assert "encoding_error" in (row.error or "")


def test_a_load_failure_leaves_a_finished_row_and_its_chunks_untouched(
    tmp_path: Path, registry: DocumentRegistry
) -> None:
    root = tmp_path / "corpus"
    write(root, "guide.md", TEXT)
    store = FakeChunkStore()
    run_pass(root, registry, store=store)
    doc_id = make_doc_id("guide.md")

    (root / "guide.md").write_bytes(BROKEN_BYTES)
    second = run_pass(root, registry, store=store)
    outcome = by_path(second)["guide.md"]

    assert outcome.is_failed
    # A terminal row cannot be nudged (§5.1); the failure is reported
    # through the outcome instead, and the old chunks stay retrievable.
    assert outcome.status is IngestStatus.INDEXED
    row = registry.get(doc_id)
    assert row is not None
    assert row.status is IngestStatus.INDEXED
    assert store.count(doc_id) > 0


def test_a_row_that_cannot_be_moved_still_reports_the_load_failure(
    tmp_path: Path, registry: DocumentRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "corpus"
    stalled = Document.create(
        source_path="guide.md",
        filename="guide.md",
        media_type="text/markdown",
        content=TEXT,
    )
    registry.create(stalled)
    write(root, "guide.md", TEXT)
    (root / "guide.md").write_bytes(BROKEN_BYTES)

    def refuse(doc_id: str, status: IngestStatus, error: str | None = None) -> None:
        raise DependencyError("registry write refused", code="dependency_unavailable")

    monkeypatch.setattr(registry, "update_status", refuse)

    result = run_pass(root, registry)
    outcome = by_path(result)["guide.md"]

    # The load failure is the actionable fact; the row keeps its previous
    # state rather than the outcome inventing a new one.
    assert outcome.is_failed
    assert outcome.code == "encoding_error"
    assert outcome.status is IngestStatus.PENDING


def test_a_registry_failure_stops_that_document_before_any_chunks(
    tmp_path: Path, registry: DocumentRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "corpus"
    write(root, "a.md", TEXT)
    write(root, "b.md", OTHER_TEXT)
    target = make_doc_id("b.md")
    real_create = registry.create

    def create(document: Document) -> None:
        if document.doc_id == target:
            raise DependencyError("registry write refused", code="dependency_unavailable")
        real_create(document)

    monkeypatch.setattr(registry, "create", create)
    store = FakeChunkStore()

    result = run_pass(root, registry, store=store)

    outcomes = by_path(result)
    assert outcomes["a.md"].is_created
    failed = outcomes["b.md"]
    assert failed.is_failed
    assert failed.code == "dependency_unavailable"
    # The row was never created, so there is no status to report.
    assert failed.status is None
    # The invariant: no chunks without a row, and the batch carried on.
    assert store.count(target) == 0
    assert registry.get(target) is None
    assert set(store.data) == {make_doc_id("a.md")}


def test_a_status_write_failure_is_reported_and_never_claims_indexed(
    tmp_path: Path, registry: DocumentRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "corpus"
    write(root, "guide.md", TEXT)
    store = FakeChunkStore()
    real = registry.update_status

    def fail_when_indexing(doc_id: str, status: IngestStatus, error: str | None = None) -> None:
        if status is IngestStatus.INDEXED:
            raise DependencyError("registry went offline", code="dependency_unavailable")
        real(doc_id, status, error)

    monkeypatch.setattr(registry, "update_status", fail_when_indexing)

    result = run_pass(root, registry, store=store)
    outcome = by_path(result)["guide.md"]

    # The chunks landed, but the confirmation did not — so the outcome
    # reports the registry's failure and the row's true state, never INDEXED,
    # while still counting what actually reached the store.
    assert outcome.is_failed
    assert outcome.code == "dependency_unavailable"
    assert outcome.status is IngestStatus.INDEXING
    assert store.count() > 0
    assert outcome.chunk_count == store.count()

    row = registry.get(make_doc_id("guide.md"))
    assert row is not None
    assert row.status is IngestStatus.INDEXING


def test_a_row_that_cannot_be_read_at_all_reports_no_status(
    tmp_path: Path, registry: DocumentRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "corpus"
    write(root, "guide.md", TEXT)

    def refuse_get(doc_id: str) -> Document | None:
        raise DependencyError("registry read refused", code="dependency_unavailable")

    def refuse_indexed(doc_id: str, status: IngestStatus, error: str | None = None) -> None:
        if status is IngestStatus.INDEXED:
            raise DependencyError("registry went offline", code="dependency_unavailable")
        # The real update_status reads the row through get() too, so the
        # walk itself must fail once reads fail — as it would in reality.
        raise DependencyError("registry write refused", code="dependency_unavailable")

    monkeypatch.setattr(registry, "get", refuse_get)
    monkeypatch.setattr(registry, "update_status", refuse_indexed)

    result = run_pass(root, registry)
    outcome = by_path(result)["guide.md"]

    assert outcome.is_failed
    assert outcome.code == "dependency_unavailable"
    assert outcome.status is None


# --- Skips: files that need nothing ------------------------------------------


def test_an_empty_file_is_skipped_without_a_row(tmp_path: Path, registry: DocumentRegistry) -> None:
    root = tmp_path / "corpus"
    write(root, "empty.md", "")

    result = run_pass(root, registry)
    outcome = by_path(result)["empty.md"]

    assert outcome.is_skipped
    assert outcome.code == EMPTY_FILE_CODE
    assert outcome.status is None
    assert registry.get(make_doc_id("empty.md")) is None


def test_a_document_the_chunker_cannot_split_is_retired_as_failed(
    tmp_path: Path, registry: DocumentRegistry
) -> None:
    """no_chunks has no indexable content — the row still ends terminal."""
    root = tmp_path / "corpus"
    write(root, "guide.md", TEXT)
    store = FakeChunkStore()

    result = run_pass(root, registry, store=store, chunker=EmptyChunker())
    outcome = by_path(result)["guide.md"]

    assert outcome.is_skipped
    assert outcome.code == NO_CHUNKS_CODE
    assert outcome.status is IngestStatus.FAILED
    assert store.count() == 0

    row = registry.get(outcome.doc_id)
    assert row is not None
    assert row.status is IngestStatus.FAILED
    assert "no_chunks" in (row.error or "")


def test_recursive_false_indexes_only_the_top_level(
    tmp_path: Path, registry: DocumentRegistry
) -> None:
    root = tmp_path / "corpus"
    write(root, "top.md", TEXT)
    write(root, "nested/deep.md", OTHER_TEXT)
    store = FakeChunkStore()

    shallow = run_pass(root, registry, store=store, recursive=False)
    assert [outcome.file.relative_path for outcome in shallow.outcomes] == ["top.md"]
    assert registry.get(make_doc_id("nested/deep.md")) is None

    deep = run_pass(root, registry, store=store, recursive=True)
    outcomes = by_path(deep)
    assert outcomes["top.md"].is_skipped
    assert outcomes["top.md"].code == UNCHANGED_CODE
    assert outcomes["nested/deep.md"].is_created


# --- Reporting: what a summary may contain -----------------------------------


def test_summaries_carry_counts_and_never_document_text(
    tmp_path: Path, registry: DocumentRegistry
) -> None:
    root = tmp_path / "corpus"
    write(root, "guide.md", CANARY)
    (root / "broken.md").write_bytes(BROKEN_BYTES)

    result = run_pass(root, registry)
    rendered = str(result.summary())

    assert MARKER not in rendered
    assert str(tmp_path) not in rendered
    assert result.summary()["created"] == 1

    created = by_path(result)["guide.md"].summary()
    assert MARKER not in str(created)
    # A file that reached classification carries its dedup kind…
    assert "kind" in created

    # …while a failure explains itself with its stable code, and a file
    # that never reached classification carries no kind at all.
    failed = by_path(result)["broken.md"]
    summary = failed.summary()
    assert summary["code"] == "encoding_error"
    # The action is the verdict; the status stays null because this file
    # never had a row to put a status on.
    assert summary["action"] == "failed"
    assert summary["status"] is None
    assert MARKER not in str(summary)
    assert "kind" not in summary


def test_the_error_line_is_always_non_empty_and_readable() -> None:
    """The registry's error column has one shape and never goes blank."""
    assert _error_line("parse_error", "file is encrypted") == "parse_error: file is encrypted"
    assert _error_line("parse_error", None) == "parse_error"
    assert _error_line(None, "something went wrong") == "something went wrong"
    assert _error_line(None, None) == "indexing failed"


def test_the_registry_view_walks_every_page(
    registry: DocumentRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A classification view missing rows past page one would re-index them."""
    rows = [
        Document.create(
            source_path=f"doc-{index}.md",
            filename=f"doc-{index}.md",
            media_type="text/markdown",
            content=f"content number {index}",
        )
        for index in range(1200)
    ]

    def paged(*, status: object = None, limit: int = 50, offset: int = 0) -> list[Document]:
        assert status is None, "the pass never filters the snapshot by status"
        return rows[offset : offset + limit]

    monkeypatch.setattr(registry, "list", paged)

    records = _all_records(registry)

    assert len(records) == 1200
    assert records[0].doc_id == rows[0].doc_id
    assert records[-1].doc_id == rows[-1].doc_id


# --- Caller errors: what must raise instead of reporting ---------------------


def test_a_missing_root_is_a_caller_error_not_a_batch(
    tmp_path: Path, registry: DocumentRegistry
) -> None:
    with pytest.raises(NotFoundError) as excinfo:
        run_pass(tmp_path / "does-not-exist", registry)
    assert excinfo.value.code == "discovery_root_not_found"


def test_a_registry_that_cannot_be_listed_stops_the_pass(
    tmp_path: Path, registry: DocumentRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "corpus"
    write(root, "guide.md", TEXT)

    def refuse(**kwargs: object) -> list[Document]:
        raise DependencyError("database is locked", code="dependency_unavailable")

    monkeypatch.setattr(registry, "list", refuse)

    with pytest.raises(DependencyError) as excinfo:
        run_pass(root, registry)
    assert excinfo.value.code == "dependency_unavailable"


# --- The real store: structural fit, end to end ------------------------------


def test_the_pass_round_trips_through_a_real_chroma_index(
    tmp_path: Path, registry: DocumentRegistry
) -> None:
    root = tmp_path / "corpus"
    write(root, "guide.md", TEXT)
    index = ChromaIndex(persist_directory=tmp_path / "chroma")
    doc_id = make_doc_id("guide.md")

    first = run_pass(root, registry, store=index)
    written = index.count()
    assert written == first.chunks_written > 0

    second = run_pass(root, registry, store=index)
    assert by_path(second)["guide.md"].is_skipped
    # Nothing re-written: the count is identical after the second pass.
    assert index.count() == written

    row = registry.get(doc_id)
    assert row is not None
    assert row.status is IngestStatus.INDEXED


def test_the_public_names_are_exported_for_the_command_to_import() -> None:
    """Stage 29's command imports exactly these names."""
    from app.ingestion import indexer

    assert indexer.index_directory is index_directory
    assert set(indexer.__all__) == {"IndexOutcome", "IndexResult", "index_directory"}
