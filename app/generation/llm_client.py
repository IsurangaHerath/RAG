"""LLM adapter wrapper — Stage 37 (`generation/llm_client.py`).

Not a direct OpenAI import exposed to the rest of the app: callers ask
for `LLMClient`, which builds its client from `Settings` (the same
`openai_api_key` check the embedder uses: key verified *before* any
client exists, so a missing key fails with `ConfigurationError`
(`openai_key_missing`) rather than a 401 from the API). Errors are mapped
into the shared hierarchy (`ProviderAuthenticationError`,
`ProviderRateLimitError`, `DependencyError` `dependency_unavailable`)
so one alert rule catches provider failures wherever they appear (§5.15,
§5.3 error-policy consistency).

The adapter does not embed — embedding lives in `app.embeddings` (§5.13-
section 5.15) - and does not retrieve; it only turns `GroundedContext` (§5.4,
Stage 36) into a `list[BaseMessage]` through the prompt layer (§5.5,
`prompts.py`, coming with `PromptBuilder`). The separation is deliberate:
changing the LLM provider is an import change at this file (`LLMClient`),
not a refactor across generation, retrieval and ingestion.
"""

from __future__ import annotations

from typing import Protocol

from langchain_chat_models import ChatOpenAI  # type: ignore[import-not-found]
from langchain_core.messages import BaseMessage

from app.config import Settings
from app.domain.errors import (
    DependencyError,
    ProviderAuthenticationError,
    ProviderRateLimitError,
)

__all__ = ["LLMClient", "LLMClientProtocol"]


class LLMClientProtocol(Protocol):
    """Structural contract: anything with these methods is an LLM adapter."""

    def generate_messages(
        self,
        messages: list[BaseMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> list[BaseMessage]: ...


class LLMClient:
    """Concrete adapter built from settings; errors mapped before escaping.

    The `ChatOpenAI` instance is created lazily (first `generate_messages`
    call) and only after `require_openai_key()` succeeds — the same
    pattern `OpenAIEmbedder.from_settings()` uses (§5.13), so a missing
    key always raises `ConfigurationError` before any network request.

    Provider errors (`openai.APIConnectionError`, `openai.APIStatusError`,
    `openai.APITimeoutError`) are translated into the shared error
    hierarchy with the original message preserved in `detail` (never in
    `message`, so no credential leaks through logging). Rate limits
    (`429`) become `ProviderRateLimitError`; connection failures become
    `DependencyError` (`dependency_unavailable`); auth failures (`401`)
    become `ProviderAuthenticationError`. This is the same family mapping
    `openai_embedder.py` (§5.13) and `chroma.py` (§5.3.1) follow.

    Temperature defaults to `0` for determinism; `stream` is not
    supported in v1 (the API route that will call this can add it
    later without changing the adapter interface).
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()
        self._client: ChatOpenAI | None = None

    def _require_client(self) -> ChatOpenAI:
        if self._client is None:
            try:
                self._client = ChatOpenAI(
                    model=self._settings.chat_model,
                    api_key=self._settings.require_openai_key(),
                    temperature=0,
                    max_tokens=2048,  # default budget, overridable per call
                    timeout=self._settings.request_timeout_seconds,
                )
            except Exception as exc:
                # Any failure during client construction (bad model name,
                # unreachable endpoint) surfaces as configuration/dependency
                # trouble, not an untyped crash.
                from openai import APIConnectionError, APIStatusError

                if isinstance(exc, APIStatusError):
                    status = getattr(exc, "status_code", 0)
                    if status == 401:
                        raise ProviderAuthenticationError(
                            "OpenAI authentication failed; check OPENAI_API_KEY",
                            code="provider_authentication_failed",
                            detail=str(exc),
                        ) from exc
                    if status == 429:
                        raise ProviderRateLimitError(
                            "OpenAI rate limit exceeded",
                            code="provider_rate_limit",
                            detail=str(exc),
                        ) from exc
                    raise DependencyError(
                        "OpenAI provider returned an unexpected status",
                        code="dependency_unavailable",
                        detail=f"status={status}: {exc}",
                    ) from exc
                if isinstance(exc, APIConnectionError) or isinstance(exc, APIConnectionError):
                    raise DependencyError(
                        "Could not connect to the OpenAI provider",
                        code="dependency_unavailable",
                        detail=str(exc),
                    ) from exc
                raise DependencyError(
                    "Failed to initialize the OpenAI client",
                    code="dependency_unavailable",
                    detail=f"{type(exc).__name__}: {exc}",
                ) from exc
        return self._client

    def generate_messages(
        self,
        messages: list[BaseMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> list[BaseMessage]:
        """Return the assistant's response messages (usually one `AIMessage`).

        Raises
        ------
        ProviderAuthenticationError
            The key was rejected (`401`).
        ProviderRateLimitError
            Rate limit hit (`429`).
        DependencyError
            Any other provider or network failure (`connection`, `timeout`,
            unexpected status, bad model name).
        """
        client = self._require_client()
        try:
            result = client.invoke(messages, temperature=temperature, max_tokens=max_tokens)
            return [result] if not isinstance(result, list) else result
        except Exception as exc:
            from openai import APIConnectionError, APIStatusError

            if isinstance(exc, APIStatusError):
                status = getattr(exc, "status_code", 0)
                if status == 401:
                    raise ProviderAuthenticationError(
                        "OpenAI authentication failed",
                        code="provider_authentication_failed",
                        detail=str(exc),
                    ) from exc
                if status == 429:
                    raise ProviderRateLimitError(
                        "OpenAI rate limit exceeded",
                        code="provider_rate_limit",
                        detail=str(exc),
                    ) from exc
                raise DependencyError(
                    "OpenAI provider error",
                    code="dependency_unavailable",
                    detail=f"status={status}: {exc}",
                ) from exc
            if isinstance(exc, APIConnectionError):
                raise DependencyError(
                    "OpenAI connection error",
                    code="dependency_unavailable",
                    detail=str(exc),
                ) from exc
            raise DependencyError(
                "Unexpected failure from the OpenAI provider",
                code="dependency_unavailable",
                detail=f"{type(exc).__name__}: {exc}",
            ) from exc
