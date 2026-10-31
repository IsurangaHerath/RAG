"""Tests for grounded prompt assembly (§5.5).

The prompt is where the grounding contract becomes instructions. These
tests pin the two things that are easy to get subtly wrong: the model must
receive the *whole* budgeted passage (not a log preview), and the citation
numbers in the prompt must be the same numbers the checker resolves later.
"""

from __future__ import annotations

import pytest

from app.domain.errors import ValidationError
from app.domain.models import GroundedContext, RetrievedChunk, make_chunk_id
from app.generation.prompts import ABSTENTION_PHRASE, GroundingPromptBuilder

DOC_ID = "doc_0123456789abcdef"
LONG_TAIL = "the settlement window closes after two business days"


def make_chunk(index: int, *, content: str, page: int | None = None) -> RetrievedChunk:
    metadata: dict[str, str | int] = {"filename": f"guide{index}.pdf"}
    if page is not None:
        metadata["page"] = page
    return RetrievedChunk(
        chunk_id=make_chunk_id(DOC_ID, index),
        doc_id=DOC_ID,
        content=content,
        distance=0.1 * (index + 1),
        metadata=metadata,
    )


def test_system_prompt_states_the_grounding_rules_and_abstention_phrase() -> None:
    chunk = make_chunk(0, content="Fees are 2%.", page=1)
    ctx = GroundedContext(query="fees", chunks=[chunk], citation_mapping={1: chunk})

    messages = GroundingPromptBuilder().build("What is the fee?", ctx)

    system = str(messages[0].content)
    assert "Answer ONLY from" in system
    assert "Every factual claim must cite" in system
    assert ABSTENTION_PHRASE in system
    assert "Never invent sources" in system


def test_user_prompt_carries_the_full_budgeted_passage_not_a_preview() -> None:
    body = "A" * 400 + " " + LONG_TAIL
    chunk = make_chunk(0, content=body, page=3)
    ctx = GroundedContext(query="settlement", chunks=[chunk], citation_mapping={1: chunk})

    messages = GroundingPromptBuilder().build("When does settlement close?", ctx)

    user = str(messages[1].content)
    assert body in user
    assert LONG_TAIL in user
    assert "guide0.pdf p.3" in user
    assert chunk.chunk_id in user


def test_citation_mapping_controls_the_prompt_numbering() -> None:
    first = make_chunk(0, content="first passage")
    second = make_chunk(1, content="second passage")
    # Deliberately non-contiguous and out of insertion order.
    ctx = GroundedContext(
        query="q",
        chunks=[first, second],
        citation_mapping={5: second, 2: first},
    )

    user = str(GroundingPromptBuilder().build("q", ctx)[1].content)

    assert user.index("[citation:2]") < user.index("[citation:5]")
    assert f"[2]: {first.chunk_id}" in user
    assert f"[5]: {second.chunk_id}" in user


def test_empty_citation_mapping_produces_an_abstention_prompt() -> None:
    ctx = GroundedContext(query="q", chunks=[], citation_mapping={})

    user = str(GroundingPromptBuilder().build("q", ctx)[1].content)

    assert "no passages retrieved" in user
    assert "[citation:" not in user
    assert "Context: (empty" in user


def test_blank_query_is_rejected_before_any_prompt_is_built() -> None:
    ctx = GroundedContext(query="q", chunks=[], citation_mapping={})

    with pytest.raises(ValidationError) as excinfo:
        GroundingPromptBuilder().build("   ", ctx)

    assert excinfo.value.code == "empty_embedding_input"


def test_history_is_inserted_before_the_current_question() -> None:
    chunk = make_chunk(0, content="Fees are 2%.")
    ctx = GroundedContext(query="fees", chunks=[chunk], citation_mapping={1: chunk})

    messages = GroundingPromptBuilder().build("What is the fee?", ctx, history=["earlier turn"])

    assert len(messages) == 3
    assert messages[1].content == "earlier turn"
    assert "What is the fee?" in str(messages[2].content)
