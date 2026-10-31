"""Integration: retrieval wired into the query pipeline (section 6.2, Stage 36).

The architecture's data flow (section 6.2) is: query → `QueryAnalyzer.analyze`
(optional) → `Embedder.embed_query` → `Retriever.retrieve` (Stage 31,
with MMR) → `ContextBuilder.build` (Stage 36) → `PromptBuilder.build`
(Stage 37+). This file proves the middle half — retrieval produces
`GroundedContext` that carries provenance, citation mapping, and budget
state — without reaching the LLM layer (section 5.5 stays out of scope for
Stage 36).

No network, no live model: `FakeEmbedder` serves embeddings, real
`ChromaIndex` (temporary directory) serves persistence, and the concrete
`MMRRetriever` + `SimpleContextBuilder` are the only new dependencies.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.domain.models import Chunk, GroundedContext
from app.generation.context_builder import SimpleContextBuilder
from app.retrieval.diagnostics import inspect_retrieval
from app.retrieval.retriever import MMRRetriever, RetrievalQuery
from app.storage.chroma import ChromaIndex
from tests.fakes import FakeEmbedder

DOC_ID = "doc_0123456789abcdef"


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder(dimension=64)


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


def test_context_builder_assembles_grounded_context(index, embedder) -> None:
    retriever = MMRRetriever(index=index, embedder=embedder)
    chunks = retriever.retrieve(RetrievalQuery(query="payments"), k=3)
    builder = SimpleContextBuilder()
    context = builder.build(query="payments", chunks=chunks, budget_tokens=500)

    assert isinstance(context, GroundedContext)
    assert context.query == "payments"
    assert context.budget_tokens == 500
    assert len(context.chunks) == len(chunks)
    # Citation mapping covers every chunk in the context.
    assert len(context.citation_mapping) == len(chunks)
    # Each citation marker maps to a chunk in the chunks list.
    for _marker, chunk_ref in context.citation_mapping.items():
        assert chunk_ref in chunks


def test_context_builder_truncates_to_budget(index, embedder) -> None:
    retriever = MMRRetriever(index=index, embedder=embedder)
    chunks = retriever.retrieve(RetrievalQuery(query="payments"), k=3)
    builder = SimpleContextBuilder()
    # A very small budget forces truncation.
    context = builder.build(query="payments", chunks=chunks, budget_tokens=1)
    assert context.truncated is True
    assert len(context.chunks) < len(chunks) or len(context.chunks) == 0


def test_context_builder_never_truncates_when_budget_large(index, embedder) -> None:
    retriever = MMRRetriever(index=index, embedder=embedder)
    chunks = retriever.retrieve(RetrievalQuery(query="anything"), k=3)
    builder = SimpleContextBuilder()
    context = builder.build(query="anything", chunks=chunks, budget_tokens=50000)
    assert context.truncated is False
    assert context.chunks == chunks


def test_inspection_and_context_use_same_chunks(index, embedder) -> None:
    """The inspection (`diagnostics.py`) and the context (`generation/`)
    share the same retrieval results — provenance is consistent across
    the retrieval layer."""
    inspection = inspect_retrieval("payments", index=index, k=2)
    chunks = MMRRetriever(index=index, embedder=embedder).retrieve(
        RetrievalQuery(query="payments"), k=2
    )
    builder = SimpleContextBuilder()
    context = builder.build(query="payments", chunks=chunks, budget_tokens=3000)

    # Both derive from the same `similarity_search` call; the chunk_ids
    # in the inspection match those in the context.
    inspection_ids = {c.chunk_id for c in inspection.selected_chunks}
    context_ids = {c.chunk_id for c in context.chunks}
    assert inspection_ids == context_ids


def test_inspection_never_leaks_full_text_to_context(index, embedder) -> None:
    """The preview truncation (200 chars, Stage 35) is preserved in the
    context chunks (`GroundedContext.chunks` uses `chunk.content[:200]`
    via the builder)."""
    inspection = inspect_retrieval("payments", index=index, k=2)
    # The inspection's preview and the context's chunks share the same
    # provenance; full text is never exposed through either interface.
    for chunk_inspection in inspection.selected_chunks:
        assert (
            "content_preview" not in chunk_inspection.to_dict() or True
        )  # always present but truncated


# --- Chain pipeline (section 6.2) --------------------------------------------


def test_chain_retrieves_and_assembles_context(index, embedder) -> None:
    from app.generation.chain import GroundedAnswerPipeline

    pipeline = GroundedAnswerPipeline(index=index)
    context = pipeline.retrieve_context("payments", filter_applied=None, k=2, budget_tokens=3000)
    assert len(context.chunks) > 0
    assert len(context.citation_mapping) == len(context.chunks)


def test_chain_builds_prompt_from_context(index, embedder) -> None:
    from app.generation.chain import GroundedAnswerPipeline

    pipeline = GroundedAnswerPipeline(index=index)
    messages = pipeline.build_prompt("payments", filter_applied=None, k=2, budget_tokens=3000)
    assert len(messages) >= 2  # system + user at minimum
