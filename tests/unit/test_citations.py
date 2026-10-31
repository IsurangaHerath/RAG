"""Tests for citation parsing and grounding verification.

The prompt can ask for citations; only a check can prove they resolve.
These tests cover the mechanical contract: markers are extracted in
order, duplicates collapse, non-citations are ignored, abstention is
recognised exactly, and an invented ``[n]`` never passes as grounded.

No network, no model, no chunk content in any error or result payload.
"""

from __future__ import annotations

import pytest

from app.domain.errors import ValidationError
from app.domain.models import GroundedContext, RetrievedChunk, make_chunk_id
from app.generation.citations import (
    CitationCheck,
    GroundingChecker,
    extract_citation_markers,
    is_abstention,
    validate_citations,
)
from app.generation.prompts import ABSTENTION_PHRASE

DOC_ID = "doc_0123456789abcdef"
MARKER = "CONFIDENTIAL-MARKET-ENTRY"


def make_chunk(index: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=make_chunk_id(DOC_ID, index),
        doc_id=DOC_ID,
        content=f"passage {index}: {MARKER} applies per account",
        distance=0.1 * index,
        metadata={"filename": f"doc{index}.pdf", "page": index + 1},
    )


@pytest.fixture
def context() -> GroundedContext:
    chunks = [make_chunk(0), make_chunk(1), make_chunk(2)]
    return GroundedContext(
        query="what applies per account?",
        chunks=chunks,
        citation_mapping={1: chunks[0], 2: chunks[1], 3: chunks[2]},
        budget_tokens=500,
        truncated=False,
    )


# --- Marker extraction ----------------------------------------------------


def test_extracts_a_single_marker() -> None:
    assert extract_citation_markers("Settlement takes one day [1].") == (1,)


def test_extracts_markers_in_first_appearance_order() -> None:
    answer = "First [3], then [1], and finally [2]."
    assert extract_citation_markers(answer) == (3, 1, 2)


def test_duplicate_markers_collapse() -> None:
    answer = "[1] says this, and [1] repeats it; [2] adds detail."
    assert extract_citation_markers(answer) == (1, 2)


def test_adjacent_markers_are_both_found() -> None:
    assert extract_citation_markers("Two sources agree [1][2].") == (1, 2)


def test_non_numeric_brackets_are_ignored() -> None:
    answer = "The tag [draft] and [note] are not citations, but [2] is."
    assert extract_citation_markers(answer) == (2,)


def test_no_citations_is_an_empty_tuple() -> None:
    assert extract_citation_markers("A plain answer with no sources.") == ()


def test_empty_answer_is_rejected_by_the_parser() -> None:
    with pytest.raises(ValidationError) as excinfo:
        extract_citation_markers("   ")
    assert excinfo.value.code == "empty_answer"


def test_non_string_answer_is_rejected_by_the_parser() -> None:
    with pytest.raises(ValidationError) as excinfo:
        extract_citation_markers(None)  # type: ignore[arg-type]
    assert excinfo.value.code == "invalid_answer"


# --- Abstention -----------------------------------------------------------


def test_exact_abstention_phrase_is_recognised() -> None:
    assert is_abstention(ABSTENTION_PHRASE) is True


def test_abstention_survives_whitespace_and_case_noise() -> None:
    noisy = f"  {ABSTENTION_PHRASE.upper().replace(' ', '   ')}  "
    assert is_abstention(noisy) is True


def test_abstention_phrase_inside_invented_prose_is_not_an_abstention() -> None:
    answer = f"{ABSTENTION_PHRASE} However, the fee is 3% per transfer [1]."
    assert is_abstention(answer) is False


# --- Grounding check ------------------------------------------------------


def test_answer_with_resolving_citations_is_grounded(context: GroundedContext) -> None:
    result = GroundingChecker().inspect("Settlement takes one day [1].", context)
    assert isinstance(result, CitationCheck)
    assert result.grounded is True
    assert result.reason is None
    assert result.markers == (1,)
    assert result.cited_chunk_ids == (context.citation_mapping[1].chunk_id,)


def test_multiple_citations_map_to_their_chunks(context: GroundedContext) -> None:
    result = GroundingChecker().inspect("Both hold [1][3].", context)
    assert result.grounded is True
    assert result.cited_chunk_ids == (
        context.citation_mapping[1].chunk_id,
        context.citation_mapping[3].chunk_id,
    )


def test_unknown_marker_blocks_grounding(context: GroundedContext) -> None:
    result = GroundingChecker().inspect("Invented evidence [9].", context)
    assert result.grounded is False
    assert result.reason == "unknown_citation"
    assert result.unknown_markers == (9,)
    assert result.cited_chunk_ids == ()


def test_known_and_unknown_markers_together_are_still_not_grounded(
    context: GroundedContext,
) -> None:
    result = GroundingChecker().inspect("Real [1] and invented [7].", context)
    assert result.grounded is False
    assert result.reason == "unknown_citation"
    assert result.unknown_markers == (7,)
    assert result.cited_chunk_ids == (context.citation_mapping[1].chunk_id,)


def test_answer_without_citations_is_not_grounded(context: GroundedContext) -> None:
    result = GroundingChecker().inspect("A confident but unsourced claim.", context)
    assert result.grounded is False
    assert result.reason == "missing_citations"
    assert result.markers == ()


def test_valid_abstention_is_not_grounded_but_is_flagged(context: GroundedContext) -> None:
    result = GroundingChecker().inspect(ABSTENTION_PHRASE, context)
    assert result.grounded is False
    assert result.abstained is True
    assert result.reason == "abstained"


def test_bool_check_matches_the_detailed_result(context: GroundedContext) -> None:
    checker = GroundingChecker()
    assert checker.check("Sourced [2].", context) is True
    assert checker.check("Unsourced.", context) is False


def test_zero_is_not_a_valid_citation_marker(context: GroundedContext) -> None:
    result = GroundingChecker().inspect("Zero [0].", context)
    assert result.grounded is False
    assert result.unknown_markers == (0,)


def test_marker_behind_empty_context_never_resolves() -> None:
    empty = GroundedContext(query="q", chunks=[], citation_mapping={})
    result = GroundingChecker().inspect("Claim [1].", empty)
    assert result.grounded is False
    assert result.unknown_markers == (1,)


def test_empty_answer_is_rejected_by_the_checker(context: GroundedContext) -> None:
    with pytest.raises(ValidationError) as excinfo:
        GroundingChecker().check("  ", context)
    assert excinfo.value.code == "empty_answer"


# --- Strict validation ----------------------------------------------------


def test_validate_returns_the_check_when_citations_resolve(context: GroundedContext) -> None:
    result = validate_citations("Sourced [1].", context)
    assert result.grounded is True


def test_validate_raises_for_an_unknown_citation(context: GroundedContext) -> None:
    with pytest.raises(ValidationError) as excinfo:
        validate_citations("Invented [12].", context)
    error = excinfo.value
    assert error.code == "invalid_citation"
    assert error.http_status == 400
    assert error.detail == "unknown citation markers: [12]"


def test_validate_allows_an_abstention(context: GroundedContext) -> None:
    assert validate_citations(ABSTENTION_PHRASE, context).abstained is True


def test_validate_allows_an_uncited_answer_but_flags_it(context: GroundedContext) -> None:
    result = validate_citations("No sources here.", context)
    assert result.grounded is False
    assert result.reason == "missing_citations"


# --- Redaction ------------------------------------------------------------


def test_failure_details_never_quote_chunk_content(context: GroundedContext) -> None:
    with pytest.raises(ValidationError) as excinfo:
        validate_citations("Invented [9].", context)
    assert MARKER not in str(excinfo.value)
    assert MARKER not in (excinfo.value.detail or "")


def test_result_payload_has_no_answer_text(context: GroundedContext) -> None:
    answer = f"Secret claim about {MARKER} [1]."
    payload = GroundingChecker().inspect(answer, context).to_dict()
    assert MARKER not in str(payload)
    assert payload["reason"] is None
    assert payload["grounded"] is True
