"""LLM adapter wrapper (section 5.5, `generation/llm_client.py`).

Not a direct OpenAI import exposed to the rest of the app: callers ask
for `LLMClient`, which builds its client from `Settings` (the same
`openai_api_key` check the embedder uses: key verified *before* any
client exists, so a missing key fails with a `ConfigurationError`
rather than a 401 from the API). Errors are mapped into the shared
hierarchy (`ProviderAuthenticationError`, `ProviderRateLimitError`,
`DependencyError` `dependency_unavailable`) so one alert rule catches
provider failures wherever they appear (section 5.15, section 5.3 error-policy
consistency).

The adapter does not embed — embedding lives in `app.embeddings` — and
does not retrieve; it turns the message list built by the prompt layer
(section 5.5, `prompts.py`) into the assistant response that
`GroundedAnswerPipeline.answer()` parses. The separation is deliberate:
changing the LLM provider is an import change at this file, not a
refactor across generation, retrieval and ingestion.

**Provider messages are not copied into our errors.** An OpenAI
authentication error quotes (part of) the credential it rejected, so
only the exception class name and HTTP status survive translation. The
original exception still rides along as ``__cause__`` for anyone who
deliberately inspects it. This is the same rule the embedding adapter
follows (section 5.13).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Protocol, cast

import openai
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from app.config import Settings
from app.domain.errors import (
    DependencyError,
    ProviderAuthenticationError,
    ProviderRateLimitError,
    RAGError,
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

    def stream(
        self,
        messages: list[BaseMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> Iterator[str]: ...


class _ClientLike(Protocol):
    """The slice of ``ChatOpenAI`` this adapter needs.

    Structural, so tests can inject a stub without inheriting from a
    langchain class or patching an import.
    """

    def invoke(
        self, messages: list[BaseMessage], **kwargs: Any
    ) -> BaseMessage | list[BaseMessage]: ...


def _translate(exc: openai.OpenAIError) -> RAGError:
    """Map one provider exception onto the shared hierarchy.

    Deliberately no provider message: authentication failures can quote
    the key. Classification (status/class) is enough to debug and safe to
    log.
    """
    if isinstance(exc, openai.AuthenticationError):
        return ProviderAuthenticationError(
            "OpenAI authentication failed; check OPENAI_API_KEY",
            code="provider_authentication_failed",
            detail=f"{type(exc).__name__}: provider rejected the credential",
        )
    if isinstance(exc, openai.RateLimitError):
        return ProviderRateLimitError(
            "OpenAI rate limit exceeded",
            code="provider_rate_limit",
            detail=f"{type(exc).__name__}: status={getattr(exc, 'status_code', 429)}",
        )
    if isinstance(exc, openai.APIStatusError):
        return DependencyError(
            "OpenAI provider returned an unexpected status",
            code="dependency_unavailable",
            detail=f"{type(exc).__name__}: status={getattr(exc, 'status_code', 0)}",
        )
    if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError)):
        return DependencyError(
            "Could not connect to the OpenAI provider",
            code="dependency_unavailable",
            detail=type(exc).__name__,
        )
    return DependencyError(
        "Unexpected failure from the OpenAI provider",
        code="dependency_unavailable",
        detail=type(exc).__name__,
    )


class LLMClient:
    """Concrete adapter built from settings; errors mapped before escaping.

    The `ChatOpenAI` instance is created lazily (first `generate_messages`
    call) and only after `require_openai_key()` succeeds — the same
    pattern `OpenAIEmbedder.from_settings()` uses (section 5.13), so a missing
    key always raises `ConfigurationError` before any network request.

    Parameters
    ----------
    settings:
        Configuration used to build the real client. Tests pass a stub
        via ``client`` instead.
    client:
        Optional pre-built delegate shaped like `_ClientLike`. Injecting
        one keeps unit tests offline and makes the adapter's behaviour
        independent of langchain's construction details.

    Temperature defaults to `0` for determinism; streaming is not part of
    this adapter yet.
    """

    def __init__(
        self, settings: Settings | None = None, *, client: _ClientLike | None = None
    ) -> None:
        self._settings = settings or Settings()
        self._client = client

    def _require_client(self) -> _ClientLike:
        """Return the delegate, building the real one at most once."""
        if self._client is None:
            # Key first: a missing credential is a configuration error,
            # not a mysterious 401 after a network round trip.
            api_key = self._settings.require_openai_key()
            try:
                self._client = cast(
                    _ClientLike,
                    ChatOpenAI(
                        model=self._settings.chat_model,
                        api_key=SecretStr(api_key),
                        temperature=0,
                        max_completion_tokens=2048,  # default budget, overridable per call
                        timeout=self._settings.request_timeout_seconds,
                    ),
                )
            except openai.OpenAIError as exc:
                raise _translate(exc) from exc
            except Exception as exc:
                # Any non-provider failure during construction (bad model
                # name, unreachable endpoint) is still dependency trouble.
                raise DependencyError(
                    "Failed to initialize the OpenAI client",
                    code="dependency_unavailable",
                    detail=type(exc).__name__,
                ) from exc
        assert self._client is not None
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
            unexpected status, bad model name), or a response that is not a
            message at all.
        """
        client = self._require_client()
        try:
            result = client.invoke(messages, temperature=temperature, max_tokens=max_tokens)
        except openai.OpenAIError as exc:
            raise _translate(exc) from exc
        except Exception as exc:
            raise DependencyError(
                "Unexpected failure from the OpenAI provider",
                code="dependency_unavailable",
                detail=type(exc).__name__,
            ) from exc

        if isinstance(result, BaseMessage):
            return [result]
        if isinstance(result, list) and all(isinstance(item, BaseMessage) for item in result):
            return result
        raise DependencyError(
            "The language model returned an unexpected response shape",
            code="llm_invalid_response",
            detail=type(result).__name__,
        )

    def stream(
        self,
        messages: list[BaseMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """Stream assistant response chunks as strings.

        Each yielded chunk is a fragment of the answer text (not a full
        message); the caller accumulates them. Provider errors are
        translated through the same mapping as `generate_messages()` and
        raised at accumulation time, so a rate limit or auth failure
        surfaces with the typed code rather than an untyped stream break.
        """
        client = self._require_client()
        try:
            # `ChatOpenAI.stream()` yields `AIMessageChunk` objects; we
            # accumulate their text content so callers receive plain
            # strings without importing langchain chunk types.
            stream_result = client.stream(
                messages,
                config={"temperature": temperature, "max_tokens": max_tokens},
            )
            for chunk_obj in stream_result:
                chunk_text: str = ""
                if hasattr(chunk_obj, "content") and isinstance(chunk_obj.content, str):
                    chunk_text = chunk_obj.content
                elif hasattr(chunk_obj, "text") and isinstance(chunk_obj.text, str):
                    chunk_text = chunk_obj.text
                elif isinstance(chunk_obj, str):
                    chunk_text = chunk_obj
                if chunk_text:
                    yield chunk_text
        except openai.OpenAIError as exc:
            raise _translate(exc) from exc
        except Exception as exc:
            raise DependencyError(
                "Unexpected failure during streaming from the OpenAI provider",
                code="dependency_unavailable",
                detail=type(exc).__name__,
            ) from exc
