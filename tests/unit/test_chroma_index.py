"""Tests for the persistent Chroma index (``app/storage/chroma.py``).

Real chromadb against a temporary directory on every test — no network,
no mock of the thing under test: persistence, the cosine collection, the
metadata schema, upsert-replaces-id semantics, the delete count, and the
translation of chroma's exceptions into the shared hierarchy. The last
test wires the pipeline from Stage 23 to this store and ingests a small
Markdown file twice, so the overwrite protocol is proven against the
real thing rather than the in-memory double.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.domain.errors import DependencyError, ValidationError
from app.domain.models import Chunk
from app.ingestion.pipeline import IngestPipeline, PipelineAction
from app.storage.chroma import COLLECTION_NAME, ChromaIndex
from tests.fakes import FakeEmbedder

DOC_ID = "doc_0123456789abcdef"
DOC_ID_B = "doc_fedcba9876543210"


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
    """Distinct, same-sized vectors — chroma fixes a dimension per collection."""
    return [[0.1 * (position + 1)] * dimension for position in range(count)]


@pytest.fixture()
def index(tmp_path: Path) -> ChromaIndex:
    """A fresh index per test, always on an explicit temporary directory."""
    return ChromaIndex(persist_directory=tmp_path / "chroma")


def stored_ids(index: ChromaIndex, doc_id: str) -> set[str]:
    """Chunk ids currently stored for one document."""
    result = index.collection.get(where={"doc_id": doc_id}, include=[])
    return set(result["ids"])


# --- The collection itself --------------------------------------------------


def test_the_collection_is_the_documented_one_with_cosine_space(
    index: ChromaIndex,
) -> None:
    """§5.3 fixes the name and the metric; the test pins both."""
    assert index.collection.name == COLLECTION_NAME
    assert index.collection.metadata == {"hnsw:space": "cosine"}


def test_data_survives_a_new_instance_on_the_same_directory(tmp_path: Path) -> None:
    """Persistence is the point: tomorrow's process finds today's index."""
    directory = tmp_path / "chroma"
    first = ChromaIndex(persist_directory=directory)
    first.upsert([chunk(0), chunk(1)], vectors(2))
    assert first.persist_directory == directory

    second = ChromaIndex(persist_directory=directory)

    assert second.count() == 2


def test_the_default_directory_comes_from_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No argument means ``CHROMA_PERSIST_DIR``, never a hard-coded path."""
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "from_env"))

    index = ChromaIndex()
    index.upsert([chunk(0)], vectors(1))

    assert (tmp_path / "from_env").exists()
    assert ChromaIndex().count() == 1


def test_an_empty_collection_name_is_rejected_before_chroma_sees_it() -> None:
    """Our own validation, with our own code, rather than whatever chroma says."""
    with pytest.raises(ValidationError) as excinfo:
        ChromaIndex(persist_directory="x", collection_name="   ")
    assert excinfo.value.code == "invalid_request"


def test_an_unopenable_directory_becomes_a_dependency_error(tmp_path: Path) -> None:
    """A path that is a file, not a directory: infrastructure trouble (502)."""
    blocker = tmp_path / "not_a_directory.txt"
    blocker.write_text("occupied", encoding="utf-8")

    with pytest.raises(DependencyError) as excinfo:
        ChromaIndex(persist_directory=blocker)
    assert excinfo.value.code == "dependency_unavailable"


# --- Writing ----------------------------------------------------------------


def test_upsert_stores_ids_text_and_the_full_metadata_schema(
    index: ChromaIndex,
) -> None:
    """§5.3's schema, chunk text included — retrieval quotes it back later."""
    index.upsert([chunk(0)], vectors(1))

    row = index.collection.get(ids=[f"{DOC_ID}:0"], include=["documents", "metadatas"])
    assert row["ids"] == [f"{DOC_ID}:0"]
    assert row["documents"] == ["chunk number 0"]
    metadata = row["metadatas"][0]
    assert metadata["doc_id"] == DOC_ID
    assert metadata["chunk_id"] == f"{DOC_ID}:0"
    assert metadata["filename"] == "api-spec.md"
    assert metadata["media_type"] == "text/markdown"
    assert metadata["section"] == "Requirements"
    stamped = datetime.fromisoformat(metadata["ingested_at"])
    assert stamped.tzinfo is not None, "ingested_at must be timezone-aware ISO"


def test_the_indexs_own_metadata_keys_cannot_be_spoofed_by_a_chunk(
    index: ChromaIndex,
) -> None:
    """``delete_document`` filters on ``doc_id``; a lie there would misdelete."""
    liar = Chunk.create(
        doc_id=DOC_ID,
        index=0,
        content="claims to be someone else",
        metadata={"doc_id": DOC_ID_B, "chunk_id": "spoofed"},
    )

    index.upsert([liar], vectors(1))

    assert stored_ids(index, DOC_ID) == {f"{DOC_ID}:0"}
    assert stored_ids(index, DOC_ID_B) == set()


def test_writing_the_same_chunk_id_replaces_rather_than_duplicates(
    index: ChromaIndex,
) -> None:
    """The store half of idempotent re-ingestion: upsert keyed by chunk_id."""
    index.upsert([chunk(0, content="first version")], vectors(1))
    index.upsert([chunk(0, content="second version")], vectors(1))

    assert index.count() == 1
    row = index.collection.get(ids=[f"{DOC_ID}:0"], include=["documents"])
    assert row["documents"] == ["second version"]


def test_writing_nothing_is_a_no_op(index: ChromaIndex) -> None:
    """An empty write writes nothing, rather than confusing chroma."""
    index.upsert([], [])
    assert index.count() == 0


def test_mismatched_chunks_and_vectors_are_rejected_before_chroma(
    index: ChromaIndex,
) -> None:
    """Chroma would raise a bare ValueError; the family owns this code."""
    with pytest.raises(DependencyError) as excinfo:
        index.upsert([chunk(0), chunk(1)], vectors(1))
    assert excinfo.value.code == "embedding_count_mismatch"


def test_a_wrong_dimension_is_reported_under_the_familys_code(
    index: ChromaIndex,
) -> None:
    """The collection's dimension is fixed by the first write."""
    index.upsert([chunk(0)], vectors(1, dimension=4))

    with pytest.raises(DependencyError) as excinfo:
        index.upsert([chunk(0, doc_id=DOC_ID_B)], vectors(1, dimension=5))
    assert excinfo.value.code == "embedding_dimension_mismatch"
    assert "dimension" in (excinfo.value.detail or "")


def test_a_storage_failure_is_translated_not_leaked(
    index: ChromaIndex, monkeypatch: pytest.MonkeyPatch
) -> None:
    """sqlite's own exception never reaches a caller in the raw."""

    def broken_upsert(**kwargs: object) -> None:
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(index, "_collection", SimpleNamespace(upsert=broken_upsert))

    with pytest.raises(DependencyError) as excinfo:
        index.upsert([chunk(0)], vectors(1))
    assert excinfo.value.code == "dependency_unavailable"
    assert "disk I/O error" in (excinfo.value.detail or "")


# --- Deleting and counting --------------------------------------------------


def test_delete_document_returns_the_removed_count_and_spares_others(
    index: ChromaIndex,
) -> None:
    """§5.3: the return value is what the pipeline reports as deleted."""
    index.upsert([chunk(0), chunk(1), chunk(2)], vectors(3))
    index.upsert([chunk(0, doc_id=DOC_ID_B)], vectors(1))

    removed = index.delete_document(DOC_ID)

    assert removed == 3
    assert stored_ids(index, DOC_ID) == set()
    assert stored_ids(index, DOC_ID_B) == {f"{DOC_ID_B}:0"}


def test_deleting_an_unknown_document_removes_nothing(index: ChromaIndex) -> None:
    """The defensive delete of a brand-new document is normal, not an error."""
    index.upsert([chunk(0)], vectors(1))

    assert index.delete_document("doc_0000000000000000") == 0
    assert index.count() == 1


def test_delete_then_upsert_leaves_only_the_new_version(index: ChromaIndex) -> None:
    """§6.1's overwrite protocol, executed against real storage."""
    index.upsert([chunk(0), chunk(1), chunk(2)], vectors(3))

    removed = index.delete_document(DOC_ID)
    index.upsert([chunk(0), chunk(1)], vectors(2))

    assert removed == 3
    assert index.count() == 2
    assert stored_ids(index, DOC_ID) == {f"{DOC_ID}:0", f"{DOC_ID}:1"}


def test_count_reflects_every_stored_chunk(index: ChromaIndex) -> None:
    """A trivial invariant the CLI's index report will rely on."""
    assert index.count() == 0
    index.upsert([chunk(0), chunk(1)], vectors(2))
    assert index.count() == 2


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


SECTIONED = (
    "# Overview\n\nThe platform exposes a payments API.\n\n"
    "# Authentication\n\nRequests carry a bearer token.\n\n"
    "# Limits\n\nBursts are smoothed per account.\n"
)


def test_the_ingest_pipeline_writes_through_the_real_index(tmp_path: Path) -> None:
    """Stage 23's pipeline against Stage 24's store: create, skip, replace."""
    source = tmp_path / "api-spec.md"
    source.write_text(SECTIONED, encoding="utf-8")
    index = ChromaIndex(persist_directory=tmp_path / "chroma")
    embedder = CountingEmbedder()
    pipeline = IngestPipeline(embedder=embedder, store=index)  # type: ignore[arg-type]

    first = pipeline.run(source, source_path="guides/api-spec.md")
    assert first.action is PipelineAction.CREATED
    assert first.chunk_count >= 2, "headings should split the file into sections"
    assert index.count() == first.chunk_count

    second = pipeline.run(source, source_path="guides/api-spec.md")
    assert second.action is PipelineAction.SKIPPED
    assert embedder.calls == 1, "an unchanged file must not be re-embedded"
    assert index.count() == first.chunk_count

    source.write_text("A single short line.\n", encoding="utf-8")
    third = pipeline.run(source, source_path="guides/api-spec.md")
    assert third.action is PipelineAction.REPLACED
    assert third.deleted_count == first.chunk_count
    assert index.count() == third.chunk_count < first.chunk_count
    old_tail = f"{first.doc_id}:{first.chunk_count - 1}"
    assert old_tail not in stored_ids(index, first.doc_id), (
        "a shortened document must not leave its old tail retrievable"
    )
