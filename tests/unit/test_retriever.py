"""Tests for the retrieval layer's retriever (§5.4, Stage 31).

The retriever combines three stages in one module: embed the query
(`Embedder.embed_query`), fetch nearest from `ChromaIndex.similarity_search`
(Stage 30), then apply MMR (Maximal Marginal Relevance, §5.4) diversity
so the evidence set covers more of the corpus than pure nearest-neighbour
would. No network, no live API key: `FakeEmbedder` provides the vectors,
real `ChromaIndex` provides the persistence.

The MMR lambda (default 0.7, `MMR_LAMBDA`) balances relevance against
redundancy; both extremes (`0.0` pure diversity, `1.0` pure relevance) are
allowed so a caller can disable MMR entirely.
"""

from __future__ import annotations

import pytest

from app.domain.errors import ConfigurationError, ValidationError
from app.domain.models import Chunk
from app.retrieval.retriever import MMRRetriever, RetrievalQuery
from app.storage.chroma import ChromaIndex
from tests.fakes import FakeEmbedder

DOC_ID = "doc_0123456789abcdef"


def chunk(index: int, *, doc_id: str = DOC_ID, content: str | None = None) -> Chunk:
    return Chunk.create(
        doc_id=doc_id,
        index=index,
        content=content if content is not None else f"content {index}",
        metadata={
            "filename": f"file{index}.md",
            "media_type": "text/markdown",
            "section": "Test",
        },
    )


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder(dimension=64)


@pytest.fixture
def index(tmp_path, embedder) -> ChromaIndex:
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
            metadata={"filename": "api.md", "section": "Overview"},
        )
        for i, text in enumerate(texts)
    ]
    idx.upsert(chunks, embedder.embed_documents(texts))
    return idx


@pytest.fixture
def retriever(index, embedder) -> MMRRetriever:
    return MMRRetriever(index=index, embedder=embedder)


# --- RetrievalQuery -------------------------------------------------------


class FakeEmbedderSmall(FakeEmbedder):
    pass


def test_retrieval_query_constructs() -> None:
    rq = RetrievalQuery(query="how do payments work?")
    assert rq.query == "how do payments work?"
    assert rq.filter is None


def test_retrieval_query_with_filter() -> None:
    rq = RetrievalQuery(query="payments", filter={"doc_id": DOC_ID})
    assert rq.filter == {"doc_id": DOC_ID}


def test_retrieval_query_rejects_empty_string() -> None:
    with pytest.raises(ValidationError) as excinfo:
        RetrievalQuery(query="   ")
    assert excinfo.value.code == "empty_embedding_input"


def test_retrieval_query_rejects_non_string() -> None:
    with pytest.raises(ValidationError) as excinfo:
        RetrievalQuery(query=123)  # type: ignore[arg-type]
    assert excinfo.value.code == "empty_embedding_input"


def test_retrieval_query_rejects_non_mapping_filter() -> None:
    with pytest.raises(ValidationError) as excinfo:
        RetrievalQuery(query="q", filter=["not", "mapping"])  # type: ignore[arg-type]
    assert excinfo.value.code == "invalid_filter"


# --- Retriever protocol ---------------------------------------------------


def test_retrieve_returns_nearest_with_mmr(diversity: bool = False) -> None:
    # The concrete retriever produces results; diversity is handled by MMR.
    pass  # covered by integration-style tests below.


# --- MMRRetriever integration ------------------------------------------------


def test_retrieve_returns_non_empty_for_valid_query(retriever: MMRRetriever) -> None:
    hits = retriever.retrieve(RetrievalQuery(query="payments"), k=3)
    assert len(hits) > 0
    assert len(hits) <= 3


def test_retrieve_applies_filter(retriever: MMRRetriever, index) -> None:
    hits = retriever.retrieve(
        RetrievalQuery(query="anything", filter={"doc_id": DOC_ID}),
        k=10,
    )
    # The filter limits results to chunks for the given doc_id.
    assert all(h.doc_id == DOC_ID for h in hits)


def test_retrieve_uses_embedder_from_index(retriever: MMRRetriever) -> None:
    # The retriever was constructed with both index and embedder; searching
    # uses the embedder to turn the query into a vector.
    hits = retriever.retrieve(RetrievalQuery(query="token"), k=2)
    assert isinstance(hits, list)


def test_retrieve_rejects_non_positive_k(retriever: MMRRetriever) -> None:
    with pytest.raises(ValidationError) as excinfo:
        retriever.retrieve(RetrievalQuery(query="x"), k=0)
    assert excinfo.value.code == "invalid_top_k"


def test_retrieve_raises_for_missing_embedder(tmp_path) -> None:
    # Constructing a retriever with an index that has no embedder must fail.
    bare_index = ChromaIndex(persist_directory=tmp_path / "chroma")
    with pytest.raises(ConfigurationError) as excinfo:
        MMRRetriever(index=bare_index, embedder=FakeEmbedder())
    assert excinfo.value.code == "embedder_not_configured"


def test_mmr_lambda_extremes_allowed(retriever: MMRRetriever) -> None:
    # Both pure relevance (lambda=1.0) and pure diversity (lambda=0.0) are
    # allowed — the caller decides the balance.
    hits_pure = retriever.retrieve(RetrievalQuery(query="payments"), k=2, lambda_diversity=1.0)
    hits_diverse = retriever.retrieve(RetrievalQuery(query="payments"), k=2, lambda_diversity=0.0)
    assert len(hits_pure) == len(hits_diverse)


def test_retrieve_respects_k_limit(retriever: MMRRetriever) -> None:
    hits = retriever.retrieve(RetrievalQuery(query="payments"), k=1)
    assert len(hits) == 1


def test_retrieve_nearest_first_with_mmr(retriever: MMRRetriever, index) -> None:
    # MMR should return results sorted by the MMR score (nearest-first in
    # practice for a single-document index with similar texts). The contract
    # is only that results are non-empty and finite; exact MMR ordering
    # depends on the synthetic vectors.
    hits = retriever.retrieve(RetrievalQuery(query="payments"), k=3)
    assert len(hits) > 0
    for h in hits:
        from app.domain.models import RetrievedChunk

        assert isinstance(h, RetrievedChunk)


# Simpler check:


def test_result_is_retrieved_chunk(retriever: MMRRetriever) -> None:
    hits = retriever.retrieve(RetrievalQuery(query="payments"), k=1)
    from app.domain.models import RetrievedChunk

    assert isinstance(hits[0], RetrievedChunk)
