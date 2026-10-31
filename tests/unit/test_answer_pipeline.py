"""End-to-end answer path on a real index, with the model stubbed (§5.5, §6.2).

The model is the one component that cannot run offline, so it is the one
component replaced: `FakeLLMClient` returns text, and the real retrieval,
context assembly, prompt construction, citation checking and `Answer`
record do the rest. These tests are the contract for the whole query path
up to the API boundary.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from app.domain.errors import (
    ConfigurationError,
    DependencyError,
    ProviderRateLimitError,
    ValidationError,
)
from app.domain.models import Answer, Chunk
from app.generation.chain import GroundedAnswerPipeline
from app.generation.prompts import ABSTENTION_PHRASE
from app.storage.chroma import ChromaIndex
from tests.fakes import FakeEmbedder, FakeLLMClient

DOC_ID = "doc_0123456789abcdef"


@pytest.fixture
def index(tmp_path: Path) -> ChromaIndex:
    embedder = FakeEmbedder(dimension=64)
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
            metadata={"filename": f"chunk{i}.md", "page": i + 1, "section": f"Sec{i}"},
        )
        for i, text in enumerate(texts)
    ]
    idx.upsert(chunks, embedder.embed_documents(texts))
    return idx


def make_pipeline(
    index: ChromaIndex, reply: str | None
) -> tuple[GroundedAnswerPipeline, FakeLLMClient]:
    fake = FakeLLMClient(reply)
    return GroundedAnswerPipeline(index=index, llm_client=fake), fake


def test_answer_resolves_citations_to_retrieved_chunks(index: ChromaIndex) -> None:
    pipeline, _ = make_pipeline(index, "Payments settle in one business day [1].")

    answer = pipeline.answer("payments", k=3)

    assert isinstance(answer, Answer)
    assert answer.grounded is True
    assert answer.abstained is False
    assert answer.reason is None
    assert len(answer.citations) == 1
    citation = answer.citations[0]
    assert citation.marker == 1
    assert citation.filename == "chunk0.md"
    assert citation.page == 1
    assert citation.chunk_id == f"{DOC_ID}:0"


def test_multiple_markers_become_multiple_citations(index: ChromaIndex) -> None:
    pipeline, _ = make_pipeline(index, "Both facts hold [1][2].")

    answer = pipeline.answer("payments", k=3)

    assert answer.grounded is True
    assert [citation.marker for citation in answer.citations] == [1, 2]
    assert len({citation.chunk_id for citation in answer.citations}) == 2


def test_an_invented_marker_never_becomes_a_verified_answer(index: ChromaIndex) -> None:
    pipeline, _ = make_pipeline(index, "Invented evidence [9].")

    with pytest.raises(ValidationError) as excinfo:
        pipeline.answer("payments", k=3)

    assert excinfo.value.code == "invalid_citation"
    assert "chunk0.md" not in str(excinfo.value)


def test_faithful_abstention_is_returned_and_flagged(index: ChromaIndex) -> None:
    pipeline, _ = make_pipeline(index, ABSTENTION_PHRASE)

    answer = pipeline.answer("payments", k=3)

    assert answer.abstained is True
    assert answer.grounded is False
    assert answer.reason == "abstained"
    assert answer.citations == ()


def test_uncited_answer_is_returned_but_not_grounded(index: ChromaIndex) -> None:
    pipeline, _ = make_pipeline(index, "A confident but unsourced claim.")

    answer = pipeline.answer("payments", k=3)

    assert answer.grounded is False
    assert answer.reason == "missing_citations"
    assert answer.citations == ()


def test_the_model_call_receives_history_temperature_and_budget(index: ChromaIndex) -> None:
    pipeline, fake = make_pipeline(index, "Sourced [1].")

    pipeline.answer(
        "payments",
        k=2,
        history=["an earlier turn"],
        temperature=0.3,
        max_tokens=256,
    )

    messages, temperature, max_tokens = fake.calls[0]
    assert temperature == 0.3
    assert max_tokens == 256
    assert "an earlier turn" in [str(message.content) for message in messages]


def test_content_blocks_are_reduced_to_their_text(index: ChromaIndex) -> None:
    block_reply = [
        AIMessage(
            content=[
                {"type": "text", "text": "Payments settle in one business day [1]."},
                {"type": "tool_call", "name": "ignored"},
            ]
        )
    ]
    fake = FakeLLMClient(block_reply)
    pipeline = GroundedAnswerPipeline(index=index, llm_client=fake)

    answer = pipeline.answer("payments", k=3)

    assert answer.text == "Payments settle in one business day [1]."
    assert answer.grounded is True


def test_a_missing_llm_is_a_configuration_error(index: ChromaIndex) -> None:
    pipeline = GroundedAnswerPipeline(index=index)

    with pytest.raises(ConfigurationError) as excinfo:
        pipeline.answer("payments", k=3)

    assert excinfo.value.code == "llm_not_configured"


def test_an_empty_model_response_is_provider_trouble(index: ChromaIndex) -> None:
    pipeline, _ = make_pipeline(index, None)

    with pytest.raises(DependencyError) as excinfo:
        pipeline.answer("payments", k=3)

    assert excinfo.value.code == "empty_llm_response"


def test_provider_failures_pass_through_unchanged(index: ChromaIndex) -> None:
    pipeline, fake = make_pipeline(index, "unused")
    failure = ProviderRateLimitError("rate limited", code="provider_rate_limit")
    fake.error = failure

    with pytest.raises(ProviderRateLimitError) as excinfo:
        pipeline.answer("payments", k=3)

    assert excinfo.value is failure


def test_invalid_top_k_is_rejected_before_generation(index: ChromaIndex) -> None:
    pipeline, fake = make_pipeline(index, "unused")

    with pytest.raises(ValidationError) as excinfo:
        pipeline.answer("payments", k=0)

    assert excinfo.value.code == "invalid_top_k"
    assert fake.calls == []
