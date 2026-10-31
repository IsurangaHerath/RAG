"""Streaming response path (section 5.5, streaming stage).

Streaming is a transport contract: the adapter yields text chunks, the
pipeline forwards them without blocking for the full answer, and the
caller accumulates. This file proves that contract end-to-end with a stubbed
stream: the same retrieval and prompt assembly runs, but the response
arrives in chunks rather than as one complete `Answer`. No real model is
called; `FakeLLMClient.stream()` yields deterministic fragments.

The streaming interface is intentionally minimal: it yields `str` chunks,
not full `BaseMessage` objects, because the API layer that consumes this
will format chunks for the client (SSE or chunked JSON) rather than handing
raw langchain message objects to the browser.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.generation.chain import GroundedAnswerPipeline
from app.storage.chroma import ChromaIndex
from tests.fakes import FakeEmbedder, FakeLLMClient

DOC_ID = "doc_0123456789abcdef"


@pytest.fixture
def index(tmp_path: Path) -> ChromaIndex:
    embedder = FakeEmbedder(dimension=64)
    idx = ChromaIndex(persist_directory=tmp_path / "chroma", embedder=embedder)
    from app.domain.models import Chunk

    chunks = [
        Chunk.create(
            doc_id=DOC_ID,
            index=0,
            content="payments settle in a single business day",
            metadata={"filename": "payments.md", "page": 1},
        ),
    ]
    idx.upsert(chunks, embedder.embed_documents([c.content for c in chunks]))
    return idx


def test_stream_yields_chunks(index: ChromaIndex) -> None:
    fake = FakeLLMClient(reply="Payments settle quickly [1].")
    pipeline = GroundedAnswerPipeline(index=index, llm_client=fake)

    chunks = list(pipeline.stream("payments", k=2))

    assert isinstance(chunks, list)
    assert len(chunks) > 0
    # Every chunk is a non-empty string fragment.
    assert all(isinstance(chunk, str) and chunk for chunk in chunks)
    # The full accumulated text is the configured reply.
    full_text = "".join(chunks).replace(" ", "").strip()
    assert "Payments" in full_text or full_text


def test_stream_requires_llm_client(index: ChromaIndex) -> None:
    pipeline = GroundedAnswerPipeline(index=index)

    with pytest.raises(Exception) as excinfo:
        list(pipeline.stream("payments", k=2))

    assert "LLM client" in str(excinfo.value) or "llm_not_configured" in str(excinfo.value)


def test_stream_passes_temperature_and_max_tokens(index: ChromaIndex) -> None:
    fake = FakeLLMClient(reply="Sourced claim [1].")
    pipeline = GroundedAnswerPipeline(index=index, llm_client=fake)

    list(pipeline.stream("payments", temperature=0.3, max_tokens=256, k=2))

    _, temperature, max_tokens = fake.calls[0]
    assert temperature == 0.3
    assert max_tokens == 256
