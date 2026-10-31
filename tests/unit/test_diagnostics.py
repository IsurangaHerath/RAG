"""Tests for the dev-only retrieval inspector (§5.4, Stage 35 — diagnostics).

Not routed through the API (§5.6), not wired into any pipeline stage,
not called by `retriever.py`. A debugging and demonstration tool only:
`inspect_retrieval(query, *, index, k, filter)` builds a
`RetrievalInspection` with provenance, raw distance, converted score,
and truncated content preview — never full text, so inspection output
can be committed or shared without a second unredacted copy of the
corpus (§5.9).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.domain.errors import DependencyError, ValidationError
from app.domain.models import Chunk
from app.retrieval.diagnostics import (
    RETRIEVAL_INSPECTION_MAX_PREVIEW,
    RetrievalInspection,
    inspect_retrieval,
)
from app.storage.chroma import ChromaIndex
from tests.fakes import FakeEmbedder

DOC_ID = "doc_0123456789abcdef"


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder(dimension=4)


@pytest.fixture
def index(tmp_path: Path, embedder) -> ChromaIndex:
    idx = ChromaIndex(persist_directory=tmp_path / "chroma", embedder=embedder)
    texts = [
        "payments settle in a single business day",
        "bursts per account are smoothed with token buckets",
        "api authentication requires a bearer token",
    ]
    chunks = [
        Chunk.create(
            doc_id=DOC_ID,
            index=i,
            content=text,
            metadata={"filename": f"chunk{i}.md", "section": f"Sec{i}"},
        )
        for i, text in enumerate(texts)
    ]
    idx.upsert(chunks, embedder.embed_documents(texts))
    return idx


# --- Basic inspection --------------------------------------------------------


def test_inspection_produces_typed_result(index, embedder) -> None:
    inspection = inspect_retrieval("payments", index=index, k=2, lambda_diversity=0.7)
    assert isinstance(inspection, RetrievalInspection)
    assert inspection.query == "payments"
    assert inspection.k_requested == 2
    assert inspection.lambda_diversity == 0.7
    assert len(inspection.selected_chunks) > 0


def test_inspection_filter_applied_is_recorded(index, embedder) -> None:
    inspection = inspect_retrieval(
        "anything",
        index=index,
        k=5,
        filter_applied={"doc_id": DOC_ID},
    )
    assert inspection.filter_applied == {"doc_id": DOC_ID}


def test_inspection_chunk_has_provenance(index, embedder) -> None:
    inspection = inspect_retrieval("payments", index=index, k=1)
    chunk_inspection = inspection.selected_chunks[0]
    assert chunk_inspection.chunk_id.startswith(DOC_ID)
    assert chunk_inspection.doc_id == DOC_ID
    assert isinstance(chunk_inspection.distance, float)
    assert isinstance(chunk_inspection.score, float)
    assert chunk_inspection.filename is not None


def test_inspection_content_is_truncated(index, embedder) -> None:
    inspection = inspect_retrieval("payments", index=index, k=1)
    preview = inspection.selected_chunks[0].content_preview
    assert len(preview) <= RETRIEVAL_INSPECTION_MAX_PREVIEW + 4  # + " ..."
    assert preview != ""  # truncated but not empty


def test_inspection_never_returns_full_content(index, embedder) -> None:
    inspection = inspect_retrieval("payments", index=index, k=1)
    docs_result = index.collection.get(
        ids=[inspection.selected_chunks[0].chunk_id], include=["documents"]
    )
    full = docs_result["documents"][0]
    # Preview truncated only when content exceeds preview limit.
    if len(full) > RETRIEVAL_INSPECTION_MAX_PREVIEW:
        assert inspection.selected_chunks[0].content_preview != full
        assert len(inspection.selected_chunks[0].content_preview) < len(full)


# --- Validation (same typed errors as retrieval layer) ----------------------


def test_inspection_rejects_empty_query(index, embedder) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        inspect_retrieval("", index=index)


def test_inspection_rejects_non_positive_k(index, embedder) -> None:
    with pytest.raises(ValueError, match="k must be"):
        inspect_retrieval("q", index=index, k=0)


def test_inspection_rejects_bad_filter(index, embedder) -> None:
    # A malformed `where` is rejected by chroma before any result is
    # returned; depending on the chroma version it surfaces either as a
    # validation error or through the storage adapter's error mapping.
    with pytest.raises((ValidationError, DependencyError)) as excinfo:
        inspect_retrieval("q", index=index, filter_applied={"$bogus": 1})
    assert excinfo.value.code in {
        "invalid_filter",
        "embedding_dimension_mismatch",
        "dependency_unavailable",
    }


def test_inspection_rejects_bad_lambda(index, embedder) -> None:
    with pytest.raises(ValueError, match="lambda"):
        inspect_retrieval("q", index=index, lambda_diversity=1.5)


def test_inspection_rejects_negative_lambda(index, embedder) -> None:
    with pytest.raises(ValueError, match="lambda"):
        inspect_retrieval("q", index=index, lambda_diversity=-0.1)


# --- Integration with score layer --------------------------------------------


def test_inspection_scores_match_distance_mapping(index, embedder) -> None:
    inspection = inspect_retrieval("payments", index=index, k=1)
    for chunk_inspection in inspection.selected_chunks:
        # The score is derived from distance through the linear mapping.
        expected = 1.0 - (chunk_inspection.distance / 2.0)
        assert chunk_inspection.score == pytest.approx(expected, abs=1e-9)


def test_inspection_index_chunk_count_matches_real_index(index, embedder) -> None:
    inspection = inspect_retrieval("anything", index=index, k=3)
    assert inspection.index_chunk_count == index.count()
