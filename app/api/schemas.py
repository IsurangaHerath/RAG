"""Pydantic request/response contracts (section 5.6, Stage 43, 53-57).

These schemas are framework-independent from the domain layer: `Answer`
(lives in `domain/models.py`) is converted to/from `AnswerResponse`
here, so the API can change its JSON shape without refactoring the
pipeline, and the domain can change its fields without breaking clients.

Redaction is preserved: `AnswerResponse` never carries the full chunk
content (`sources` only lists chunk ids and filenames, not text), and
`QueryRequest` never logs the user's question beyond the standard access
log line (`observability/logging.py`, section 5.9).

No live model is called by this module; it only validates and serialises.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

__all__ = [
    "AnswerResponse",
    "DocumentListResponse",
    "HealthResponse",
    "QueryRequest",
    "ReadyResponse",
]


class QueryRequest(BaseModel):
    """POST `/v1/query` body (section 5.6, `routes/query.py`)."""

    question: str = Field(..., min_length=1, max_length=4000)
    doc_ids: list[str] | None = None
    top_k: int = Field(default=5, ge=1, le=50)
    stream: bool = False

    @field_validator("question")
    @classmethod
    def _reject_empty_or_whitespace_only(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("question must not be empty or whitespace-only")
        return stripped


class CitationResponse(BaseModel):
    """A resolved ``[n]`` citation (section 5.1, `domain/models.Citation`)."""

    marker: int
    chunk_id: str
    doc_id: str
    filename: str | None = None
    page: int | None = None
    section: str | None = None


class SourceResponse(BaseModel):
    """A cited passage (section 5.5, redacted preview, no full chunk text)."""

    chunk_id: str
    doc_id: str
    filename: str | None
    page: int | None
    section: str | None
    preview: str = Field(..., max_length=200)


class AnswerResponse(BaseModel):
    """POST `/v1/query` response (section 5.6, `routes/query.py`)."""

    query: str
    text: str
    citations: list[CitationResponse]
    grounded: bool
    abstained: bool
    reason: str | None = None
    sources: list[SourceResponse] | None = None


class DocumentStatusResponse(BaseModel):
    """GET `/v1/documents/{id}` response (section 5.6, `routes/documents.py`)."""

    doc_id: str
    filename: str
    status: str
    chunk_count: int | None = None
    error: str | None = None


class DocumentListResponse(BaseModel):
    """GET `/v1/documents` paginated response."""

    results: list[DocumentStatusResponse]
    total: int


class HealthResponse(BaseModel):
    """GET `/healthz` (section 5.6)."""

    status: str = "ok"


class ReadyResponse(BaseModel):
    """GET `/readyz` (section 5.6); checks Chroma and OpenAI reachability."""

    status: str
    chroma: str
    openai: str
