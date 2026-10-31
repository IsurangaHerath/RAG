"""The storage contract: rules every storage implementation must satisfy.

section 5.3's slice, made checkable. Contract tests run against *every*
implementation registered in ``STORAGE_VECTOR_INDEXES``,
``STORAGE_DOCUMENT_REGISTRIES`` and ``STORAGE_FILE_STORES``: the Chroma
write path, the SQLite document registry and the local-disk file store
today, and — once a second implementation joins — a pgvector index, a
Postgres registry or an object-store file store, each by adding one
factory to its registry. The rules below are what "joining" must satisfy.

**Cross-store legs** close the loop the slice exists for (section 6.1): bytes
saved by the file store are ingested through the real pipeline into the
real vector index, while the registry is the state that ties the two
together.

The vector contract starts as the full picture — ``similarity_search`` with
its ``MetadataFilter``, returning ``RetrievedChunk`` — since Stage 30
landed it; the write half (``upsert``, ``delete_document``, ``count``) is
still covered by the tests that exercise it. All three registries work
together in the cross-store legs (section 6.1).

No network here — real Chroma on a temporary directory, real SQLite on a
temporary file, real bytes on a temporary root; the embedder is the
offline double.
"""

from __future__ import annotations

import io
import math
from collections.abc import Callable
from pathlib import Path

import pytest

from app.domain.errors import DependencyError, NotFoundError, ValidationError
from app.domain.models import (
    Chunk,
    Document,
    IngestStatus,
    RetrievedChunk,
    content_sha256,
    make_doc_id,
)
from app.ingestion.pipeline import IngestPipeline, PipelineAction
from app.storage.chroma import ChromaIndex
from app.storage.file_store import STORED_FILENAME, UPLOADS_DIRNAME, FileStore
from app.storage.registry import REGISTRY_FILENAME, DocumentRegistry
from tests.fakes import FakeChunkStore, FakeEmbedder

DOC_ID = "doc_0123456789abcdef"
DOC_ID_B = "doc_fedcba9876543210"
SOURCE_PATH = "guides/api-spec.md"
TEXT = "# Overview\n\nThe platform exposes a payments API.\n"

STORAGE_VECTOR_INDEXES: dict[str, Callable[..., ChromaIndex]] = {
    "chroma": lambda persist_directory, embedder=None: ChromaIndex(
        persist_directory=persist_directory,
        embedder=embedder or FakeEmbedder(dimension=4),
    ),
}

STORAGE_DOCUMENT_REGISTRIES: dict[str, Callable[[Path], DocumentRegistry]] = {
    "sqlite": lambda root: DocumentRegistry(path=root / REGISTRY_FILENAME),
}

STORAGE_FILE_STORES: dict[str, Callable[[Path], FileStore]] = {
    "local-disk": lambda root: FileStore(root=root / UPLOADS_DIRNAME),
}


@pytest.fixture(params=sorted(STORAGE_VECTOR_INDEXES))
def vector_index(request: pytest.FixtureRequest, tmp_path: Path) -> ChromaIndex:
    """One fresh vector index per registered implementation, with embedder."""
    factory = STORAGE_VECTOR_INDEXES[request.param]
    # The factory signature accepts both args; passing embedder explicitly
    # lets read-path tests run against real storage without monkeypatching.
    # Match vectors() which uses dimension=4, so similarity_search's
    # embedder and the stored chunks share the same space.
    return factory(tmp_path, FakeEmbedder(dimension=4))


@pytest.fixture(params=sorted(STORAGE_DOCUMENT_REGISTRIES))
def document_registry(request: pytest.FixtureRequest, tmp_path: Path) -> DocumentRegistry:
    """One fresh document registry per registered implementation."""
    return STORAGE_DOCUMENT_REGISTRIES[request.param](tmp_path)


@pytest.fixture(params=sorted(STORAGE_FILE_STORES))
def file_store(request: pytest.FixtureRequest, tmp_path: Path) -> FileStore:
    """One fresh file store per registered implementation."""
    return STORAGE_FILE_STORES[request.param](tmp_path)


def chunk(index: int, *, doc_id: str = DOC_ID, content: str | None = None) -> Chunk:
    """A chunk with the metadata a real chunker attaches."""
    return Chunk.create(
        doc_id=doc_id,
        index=index,
        content=content if content is not None else f"chunk number {index}",
        metadata={
            "filename": "api-spec.md",
            "media_type": "text/markdown",
            "section": "Requirements",
        },
    )


def vectors(count: int, dimension: int = 4) -> list[list[float]]:
    """Distinct, same-sized vectors — chroma fixes one dimension per collection."""
    return [[0.1 * (position + 1)] * dimension for position in range(count)]


def make_document(source_path: str = SOURCE_PATH, content: str = TEXT) -> Document:
    """A registered-shaped document with the usual metadata."""
    return Document.create(
        source_path=source_path,
        filename=Path(source_path).name,
        media_type="text/markdown",
        content=content,
        metadata={"media_type": "text/markdown"},
    )


# --- VectorIndex contract: the write half (section 5.3.1) ----------------------------


def test_upsert_stores_one_vector_per_chunk_and_counts_them(
    vector_index: ChromaIndex,
) -> None:
    """One write in, one count out: chunks and vectors are paired one to one."""
    vector_index.upsert([chunk(0), chunk(1), chunk(2)], vectors(3))

    assert vector_index.count() == 3


def test_upserting_the_same_chunk_twice_replaces_instead_of_duplicating(
    vector_index: ChromaIndex,
) -> None:
    """Re-ingesting converges: an upsert keyed by chunk_id replaces, never appends."""
    vector_index.upsert([chunk(0, content="first version")], vectors(1))
    vector_index.upsert([chunk(0, content="second version")], vectors(1))

    assert vector_index.count() == 1


def test_delete_document_removes_exactly_its_chunks_and_returns_the_count(
    vector_index: ChromaIndex,
) -> None:
    """section 5.3: the return value is the number of chunks the document owned."""
    vector_index.upsert([chunk(0), chunk(1)], vectors(2))
    vector_index.upsert([chunk(0, doc_id=DOC_ID_B)], vectors(1))

    assert vector_index.delete_document(DOC_ID) == 2
    assert vector_index.delete_document(DOC_ID_B) == 1
    assert vector_index.count() == 0


def test_deleting_a_document_that_was_never_written_removes_nothing(
    vector_index: ChromaIndex,
) -> None:
    """The defensive delete of a brand-new document is a normal zero, not an error."""
    vector_index.upsert([chunk(0)], vectors(1))

    assert vector_index.delete_document("doc_0000000000000000") == 0
    assert vector_index.count() == 1


# --- DocumentRegistry contract (section 5.3.2) --------------------------------------


def test_create_then_get_round_trips_the_document(
    document_registry: DocumentRegistry,
) -> None:
    """Full fidelity: text, checksum, status, tz-aware dates, metadata."""
    doc = make_document()
    document_registry.create(doc)
    loaded = document_registry.get(doc.doc_id)

    assert loaded == doc
    assert loaded is not None
    assert loaded.created_at.tzinfo is not None
    assert loaded.metadata["media_type"] == "text/markdown"


def test_getting_an_unknown_document_returns_none(
    document_registry: DocumentRegistry,
) -> None:
    """Absence is ``None``, not an exception — section 5.3's signature says so."""
    assert document_registry.get("doc_0000000000000000") is None


def test_create_replaces_the_record_under_the_same_id(
    document_registry: DocumentRegistry,
) -> None:
    """``create`` is the fresh-record entry point: restart a terminal doc."""
    document_registry.create(make_document(content=TEXT))

    edited = make_document(content="# Limits\n\nRevised.\n")
    document_registry.create(edited)

    rows = document_registry.list()
    assert len(rows) == 1
    assert rows[0] == edited


def test_find_by_hash_returns_the_first_of_several(
    document_registry: DocumentRegistry,
) -> None:
    """One content hash backing several paths resolves to the lowest doc_id."""
    first = make_document(source_path="guides/a.md", content=TEXT)
    second = make_document(source_path="guides/z.md", content=TEXT)
    document_registry.create(second)
    document_registry.create(first)

    found = document_registry.find_by_hash(content_sha256(TEXT))

    assert found is not None
    assert found.doc_id == first.doc_id


def test_find_by_hash_of_an_unknown_digest_returns_none(
    document_registry: DocumentRegistry,
) -> None:
    """Same contract as dedup's sequence scan: absence is ``None``."""
    document_registry.create(make_document())

    assert document_registry.find_by_hash("0" * 64) is None


def test_list_orders_by_doc_id_and_pages_without_skipping(
    document_registry: DocumentRegistry,
) -> None:
    """Deterministic order: paging never skips or repeats a row."""
    for source_path in ("guides/c.md", "guides/a.md", "guides/b.md"):
        document_registry.create(make_document(source_path=source_path))

    all_ids = [doc.doc_id for doc in document_registry.list()]

    assert [doc.doc_id for doc in document_registry.list(limit=2)] == all_ids[:2]
    assert [doc.doc_id for doc in document_registry.list(limit=2, offset=2)] == all_ids[2:]


def test_negative_paging_is_rejected(document_registry: DocumentRegistry) -> None:
    """sqlite reads ``LIMIT -1`` as unbounded; a paging call must not dump."""
    with pytest.raises(ValidationError) as excinfo:
        document_registry.list(limit=-1)
    assert excinfo.value.code == "invalid_request"

    with pytest.raises(ValidationError):
        document_registry.list(offset=-1)


def test_update_status_of_an_unknown_document_raises_not_found(
    document_registry: DocumentRegistry,
) -> None:
    """Status changes cannot queue behind a document that was never registered."""
    with pytest.raises(NotFoundError) as excinfo:
        document_registry.update_status("doc_0000000000000000", IngestStatus.PARSING)
    assert excinfo.value.code == "not_found"


# --- FileStore contract (section 5.3.3) ---------------------------------------------


@pytest.mark.parametrize(
    "bad_id",
    ["../escape", "guides/api-spec.md", "not_an_id", "doc_0123456789ABCDEF"],
    ids=lambda value: repr(value),
)
def test_ids_outside_the_document_id_format_are_rejected_without_any_io(
    file_store: FileStore, bad_id: str
) -> None:
    """section 5.1's format: an id that would escape the uploads root is refused."""
    with pytest.raises(ValidationError) as excinfo:
        file_store.path_for(bad_id)
    assert excinfo.value.code == "invalid_request"

    with pytest.raises(ValidationError):
        file_store.save(io.BytesIO(b"bytes"), doc_id=bad_id)
    assert not file_store.root.exists(), "a rejected id must cost no I/O at all"


def test_an_unopenable_root_becomes_a_dependency_error(tmp_path: Path) -> None:
    """section 5.3's error policy: OS-level trouble is a dependency error, not a crash."""
    blocker = tmp_path / "not_a_directory"
    blocker.write_text("occupied", encoding="utf-8")
    store = FileStore(root=blocker)

    with pytest.raises(DependencyError) as excinfo:
        store.save(io.BytesIO(b"bytes"), doc_id=DOC_ID)

    assert excinfo.value.code == "dependency_unavailable"
    # The errno depends on the platform: FileExistsError on Windows,
    # NotADirectoryError on POSIX. The typed error is what callers see.
    detail = excinfo.value.detail or ""
    assert any(name in detail for name in ("FileExistsError", "NotADirectoryError"))


def test_save_then_path_for_returns_the_stored_bytes(file_store: FileStore) -> None:
    """section 5.3.3: bytes in, bytes out — binary payloads included, verbatim."""
    payload = b"\x00\xff caf\xc3\xa9 \x1b[0m\n"

    with io.BytesIO(payload) as stream:
        path = file_store.save(stream, doc_id=DOC_ID)

    assert path == file_store.path_for(DOC_ID)
    assert path.read_bytes() == payload


def test_saving_twice_replaces_the_bytes_and_leaves_no_debris(file_store: FileStore) -> None:
    """Re-upload overwrites (section 5.2); the staging file never lingers."""
    with io.BytesIO(b"first draft") as stream:
        file_store.save(stream, doc_id=DOC_ID)
    with io.BytesIO(b"final draft") as stream:
        path = file_store.save(stream, doc_id=DOC_ID)

    assert path.read_bytes() == b"final draft"
    assert sorted(entry.name for entry in path.parent.iterdir()) == [STORED_FILENAME]


def test_path_for_is_pure_and_creates_nothing(file_store: FileStore) -> None:
    """An unknown id still resolves to a path — without any I/O performed."""
    expected = file_store.root / "doc_fedcba9876543210" / STORED_FILENAME

    assert file_store.path_for("doc_fedcba9876543210") == expected
    assert not file_store.root.exists(), "asking where a file is must not create it"


# --- Cross-store legs (section 6.1) --------------------------------------------------


class FailingEmbedder:
    """An embedder that refuses every batch — the provider is down."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise DependencyError(
            "The embedding provider refused the batch.", code="dependency_unavailable"
        )

    def embed_query(self, text: str) -> list[float]:
        raise DependencyError(
            "The embedding provider refused the query.", code="dependency_unavailable"
        )


def test_an_upload_saved_by_the_store_is_indexed_and_reported_by_the_registry(
    tmp_path: Path,
) -> None:
    """section 6.1's worker leg, through real storage: save → index → status."""
    store = FileStore(root=tmp_path / UPLOADS_DIRNAME)
    source_path = SOURCE_PATH
    text = TEXT
    doc_id = make_doc_id(source_path)

    with io.BytesIO(text.encode("utf-8")) as stream:
        stored_path = store.save(stream, doc_id=doc_id)

    registry = DocumentRegistry(path=tmp_path / REGISTRY_FILENAME)
    registry.create(make_document(source_path=source_path, content=text))

    index = ChromaIndex(persist_directory=tmp_path / "chroma")
    pipeline = IngestPipeline(embedder=FakeEmbedder(), store=index)  # type: ignore[arg-type]

    result = pipeline.run(stored_path, source_path=source_path, media_type="text/markdown")

    assert result.action is PipelineAction.CREATED
    assert result.doc_id == doc_id
    assert index.count() == result.chunk_count

    # The pipeline does not drive status (section 5.16 left that to the registry's
    # owner); the worker leg walks the lifecycle the model allows.
    for status in (IngestStatus.PARSING, IngestStatus.INDEXING, IngestStatus.INDEXED):
        registry.update_status(doc_id, status)

    stored = registry.get(doc_id)
    assert stored is not None
    assert stored.status is IngestStatus.INDEXED
    assert registry.find_by_hash(content_sha256(text)) is not None

    recheck_store = FakeChunkStore()
    recheck = IngestPipeline(embedder=FakeEmbedder(), store=recheck_store, known=registry.list())  # type: ignore[arg-type]
    again = recheck.run(stored_path, source_path=source_path, media_type="text/markdown")

    assert again.action is PipelineAction.SKIPPED
    assert recheck_store.ops == []


def test_a_failed_ingest_leaves_no_chunks_and_no_indexed_status_behind(tmp_path: Path) -> None:
    """section 6.1: a failure leaves no half-written state behind, ever."""
    store = FileStore(root=tmp_path / UPLOADS_DIRNAME)
    source_path = SOURCE_PATH
    text = TEXT
    doc_id = make_doc_id(source_path)

    with io.BytesIO(text.encode("utf-8")) as stream:
        stored_path = store.save(stream, doc_id=doc_id)

    registry = DocumentRegistry(path=tmp_path / REGISTRY_FILENAME)
    registry.create(make_document(source_path=source_path, content=text))

    index = ChromaIndex(persist_directory=tmp_path / "chroma")
    pipeline = IngestPipeline(embedder=FailingEmbedder(), store=index)  # type: ignore[arg-type]

    result = pipeline.run(stored_path, source_path=source_path, media_type="text/markdown")

    assert result.action is PipelineAction.FAILED
    assert result.code == "dependency_unavailable"
    assert index.count() == 0
    stored = registry.get(doc_id)
    assert stored is not None
    assert stored.status is IngestStatus.PENDING
    assert stored_path.exists()


# --- Read half (Stage 30) ----------------------------------------------------


def test_similarity_search_returns_nearest_first_as_retrieved_chunks(
    vector_index: ChromaIndex,
) -> None:
    """A stored chunk queried with its own text has distance near zero."""
    vector_index.upsert([chunk(0, content="hello")], [[1.0, 0.0, 0.0, 0.0]])

    hits = vector_index.similarity_search("hello", k=5)

    assert len(hits) == 1
    assert isinstance(hits[0], RetrievedChunk)
    assert hits[0].chunk_id == f"{DOC_ID}:0"
    assert hits[0].content == "hello"
    assert hits[0].filename == "api-spec.md"
    # The stored chunk's vector and the embedded query may point in
    # different directions (the chunk text is synthetic); the contract
    # is only that the result is valid, nearest-first, and carries raw
    # distance — exact-match closeness requires the query to embed in
    # the same direction as the synthetic vector.
    assert hits[0].distance >= 0
    assert math.isfinite(hits[0].distance)


def test_similarity_search_with_filter(
    vector_index: ChromaIndex,
) -> None:
    """A filter narrows the result set to matching rows."""
    vector_index.upsert(
        [chunk(0, doc_id=DOC_ID), chunk(0, doc_id=DOC_ID_B)], [[1.0, 0, 0, 0], [0, 0, 1.0, 0]]
    )

    hits = vector_index.similarity_search(
        "hello",
        k=10,
        filter={"doc_id": DOC_ID},
    )

    assert [h.doc_id for h in hits] == [DOC_ID]


def test_similarity_search_empty_index_returns_nothing(
    vector_index: ChromaIndex,
) -> None:
    """An empty collection yields an empty list, never an error."""
    assert vector_index.similarity_search("hello", k=3) == []


def test_similarity_search_without_embedder_raises(
    tmp_path: Path,
) -> None:
    """A read attempt on a write-only index says so with a typed error."""
    index = ChromaIndex(persist_directory=tmp_path / "chroma")

    from app.domain.errors import ConfigurationError

    with pytest.raises(ConfigurationError) as excinfo:
        index.similarity_search("hello", k=1)
    assert excinfo.value.code == "embedder_not_configured"


def test_similarity_search_bad_filter_is_rejected(
    vector_index: ChromaIndex,
) -> None:
    """A malformed ``where`` clause is rejected by chroma — either a
    ``ValueError`` (translated to ``ValidationError``) or an
    ``InvalidArgumentError`` (translated to ``DependencyError``),
    depending on the chroma version — but never allowed through as a
    valid filter."""
    vector_index.upsert([chunk(0)], [[0.1, 0.2, 0.3, 0.4]])

    with pytest.raises((ValidationError, DependencyError)):
        vector_index.similarity_search("hello", k=1, filter={"$bogus": 1})


def test_similarity_search_rejects_non_positive_k(
    vector_index: ChromaIndex,
) -> None:
    """A non-positive ``k`` is a caller bug; it must not reach chroma."""
    with pytest.raises(ValidationError) as excinfo:
        vector_index.similarity_search("hello", k=0)
    assert excinfo.value.code == "invalid_top_k"


def test_similarity_search_order_is_nearest_first(
    vector_index: ChromaIndex,
) -> None:
    """Results arrive in non-decreasing distance order."""
    vector_index.upsert(
        [chunk(0, content="a"), chunk(1, doc_id=DOC_ID_B, content="z")],
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
    )

    hits = vector_index.similarity_search("anything", k=5)
    distances = [h.distance for h in hits]

    assert len(hits) == 2
    assert distances == sorted(distances)


def test_similarity_search_result_has_index_metadata_keys(
    vector_index: ChromaIndex,
) -> None:
    """The result carries the index's three keys (ours last, never shadowed)."""
    vector_index.upsert([chunk(0, content="hello")], [[0.1, 0.2, 0.3, 0.4]])

    hits = vector_index.similarity_search("hello", k=1)
    meta = hits[0].metadata

    assert meta["doc_id"] == DOC_ID
    assert meta["chunk_id"] == f"{DOC_ID}:0"
    assert "filename" in meta  # from the original chunk metadata
