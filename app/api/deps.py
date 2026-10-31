"""Dependency providers (§5.6, `api/deps.py`).

These are FastAPI `Depends()` factories that inject the shared
components (`ChromaIndex`, `LLMClient`) without creating them per
request. No secrets are constructed here; the `LLMClient` builds its
delegate from `Settings` lazily, and `ChromaIndex` is created once at
app startup. If `OPENAI_API_KEY` is missing, `LLMClient.generate_messages()`
raises `ConfigurationError` (not `api/deps.py`), so routes stay clean.
"""

from __future__ import annotations

from collections.abc import Generator

from fastapi import Depends, HTTPException, Request, status

from app.config import Settings, get_settings
from app.generation.llm_client import LLMClient
from app.storage.chroma import ChromaIndex

__all__ = ["get_settings_dep", "get_chroma_index", "get_llm_client"]


def get_settings_dep() -> Settings:
    return get_settings()


def get_chroma_index(settings: Settings = Depends(get_settings_dep)) -> ChromaIndex:
    """Build or reuse the persistent Chroma index.

    In production this would be a singleton cached in application state;
    for v1 the index is rebuilt on startup, which is safe because Chroma
    reads its persistent directory and the embedder is stateless.
    """
    try:
        from app.embeddings.openai_embedder import OpenAIEmbedder

        embedder = OpenAIEmbedder.from_settings(settings)
    except Exception as exc:
        # A missing key surfaces here as `ConfigurationError`; any other
        # failure is dependency trouble (`dependency_unavailable`).
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Embedding service unavailable: {type(exc).__name__}",
        ) from exc

    return ChromaIndex(
        persist_directory=settings.chroma_persist_directory,
        embedder=embedder,
    )


def get_llm_client(settings: Settings = Depends(get_settings_dep)) -> LLMClient:
    """Build the LLM adapter from settings; fails lazily on missing key."""
    return LLMClient(settings=settings)
