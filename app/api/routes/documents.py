"""Document management endpoints (section 5.6, `routes/documents.py`).

These routes are thin transport layers over the ingestion service and
storage layer (section 5.2, section 5.3). They never load file bytes themselves; the
service (`app/ingestion/service.py`) owns discovery, load and metadata,
so a future move from SQLite to PostgreSQL requires no change to these
routes.

The list endpoint (`GET /v1/documents`) returns `DocumentStatusResponse`
records, which omit `content` and `content_sha256` by design (section 5.1,
`Document.summary()`), so uploading a PDF never leaks its text through
an unauthenticated list call.
"""

from fastapi import APIRouter

router = APIRouter(prefix="/v1/documents", tags=["documents"])


@router.get("")
async def list_documents() -> dict:
    """Placeholder: real implementation delegates to registry (section 5.3.2)."""
    return {"results": [], "total": 0}
