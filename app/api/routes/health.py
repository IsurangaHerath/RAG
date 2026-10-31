"""Health and readiness endpoints (§5.6, `routes/health.py`).

`/healthz` is a liveness check — the process is running.
`/readyz` checks that the two dependencies (`ChromaIndex`, `LLMClient`)
are configured. A missing `OPENAI_API_KEY` makes `LLMClient` unavailable,
so readiness fails with `503`; the health endpoint stays `200` because
liveness does not require external connectivity (§5.6, architecture).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_chroma_index, get_llm_client
from app.api.schemas import HealthResponse, ReadyResponse
from app.generation.llm_client import LLMClient

router = APIRouter(tags=["health"])


@router.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    """Liveness: the process is alive (§5.6)."""
    return HealthResponse(status="ok")


@router.get("/readyz", response_model=ReadyResponse)
async def readyz(
    index=Depends(get_chroma_index),
    llm_client=Depends(get_llm_client),
) -> ReadyResponse:
    """Readiness: Chroma index and LLM adapter are configured (§5.6).

    A missing `OPENAI_API_KEY` makes the LLM adapter unavailable,
    which surfaces as `openai: unavailable` rather than `500`, so a load
    balancer can distinguish "not ready" from a crash.
    """
    chroma_status = "ready" if index is not None else "unavailable"
    openai_status = "ready" if isinstance(llm_client, LLMClient) else "unavailable"
    status_code = "ready" if chroma_status == "ready" and openai_status == "ready" else "not_ready"
    return ReadyResponse(
        status=status_code,
        chroma=chroma_status,
        openai=openai_status,
    )
