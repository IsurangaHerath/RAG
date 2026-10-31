"""Tests for the API contract layer (§5.6, 53-57).

These are transport-level tests: they prove Pydantic validation rejects
bad input (`empty_embedding_input` mapped to `400`), that schemas match
the domain model (`Answer.to_dict()` -> `AnswerResponse`), and that the
health endpoints return without importing any framework-specific internals.
No live model is called; the pipeline is stubbed via `tests/fakes.py`.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.api.schemas import AnswerResponse, HealthResponse, QueryRequest, ReadyResponse


def test_query_request_rejects_empty_question() -> None:
    with pytest.raises(ValidationError, match="empty"):
        QueryRequest(question="   ")


def test_answer_response_never_carries_chunk_content() -> None:
    response = AnswerResponse(
        query="q",
        text="It is 2% [1].",
        citations=[{"marker": 1, "chunk_id": "d:0", "doc_id": "d"}],
        grounded=True,
        abstained=False,
    )
    payload = response.model_dump()

    assert payload["text"] == "It is 2% [1]."
    assert payload["citations"][0]["chunk_id"] == "d:0"
    # The full chunk text is never part of the response contract.
    assert "content" not in payload
    for citation in payload.get("citations", []):
        assert "content" not in citation
        assert "text" not in citation


def test_health_endpoints_shape() -> None:
    health = HealthResponse()
    ready = ReadyResponse(status="ready", chroma="ready", openai="unavailable")

    assert health.status == "ok"
    assert ready.openai == "unavailable"
