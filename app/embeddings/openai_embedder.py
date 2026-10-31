"""The OpenAI adapter: our ``Embedder`` contract over langchain's client (Stage 20).

The contract exists (section 5.12 of ARCHITECTURE); this module is the first
implementation of it, and it does three jobs around
``langchain_openai.OpenAIEmbeddings``:

1. **Validate input** with the shared rule from ``base.py`` — blank text
   is a caller bug, answered locally with ``empty_embedding_input``
   rather than paid for with a round trip.
2. **Delegate in batches, through one retry policy.** ``iter_batches``
   splits the texts (128 per request by default) and ``with_retries``
   wraps every call — both supplied by ``batching.py`` (Stage 21), which
   owns the policy end to end: the transport itself stays at
   ``max_retries=0`` so no second, invisible policy can stack on top
   (three retries behind three retries means nine requests, and nobody
   can reason about the worst case).
3. **Validate output and translate failures.** The provider's response is
   checked before any caller sees it (count, dimension consistency,
   finite components), and provider exceptions become our error
   hierarchy, so the rest of the application catches ``RAGError`` and
   nothing else.

**Why the provider's message never comes with it.** OpenAI's
authentication errors quote (part of) the key they rejected — copying
that text into our ``message`` or ``detail`` would put a credential into
logs and API responses, which ``errors.py`` forbids outright. Only the
exception class name and HTTP status survive the trip: enough to debug
with, safe to log. The original exception still rides along as
``__cause__``, so full context exists for anyone who catches and
inspects it deliberately.

Why ``OpenAIEmbeddings`` and not the raw ``openai`` client: section 7.3 pinned
``langchain-openai`` for exactly this class (and ``ChatOpenAI`` later for
generation), it already ships in the pinned dependencies, and it handles
the awkward parts — token-length splitting of long inputs, key handling —
for free. The adapter keeps its type out of every signature except its
own construction.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from functools import partial
from typing import Protocol, TypeVar

import openai
from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr

from app.config import Settings
from app.domain.errors import (
    DependencyError,
    ProviderAuthenticationError,
    ProviderRateLimitError,
    RAGError,
)
from app.embeddings.base import require_embedding_text
from app.embeddings.batching import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_RETRY_WAIT,
    WaitBaseT,
    iter_batches,
    require_batch_size,
    with_retries,
)

__all__ = ["OpenAIEmbedder"]

T = TypeVar("T")


class _ClientLike(Protocol):
    """What the adapter needs from its client — langchain's shape, not ours.

    ``OpenAIEmbeddings.embed_documents`` takes ``list[str]`` where
    ``Embedder`` promises ``Sequence[str]``, so the two protocols cannot
    be the same type; the adapter bridges the difference (validation
    builds a list anyway). Structural typing means a stub in tests needs
    no inheritance, and ``mypy`` checks the real class against this at
    the single construction site in ``from_settings``.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...  # pragma: no cover

    def embed_query(self, text: str) -> list[float]: ...  # pragma: no cover


class OpenAIEmbedder:
    """``Embedder`` backed by OpenAI through ``langchain_openai.OpenAIEmbeddings``.

    Parameters
    ----------
    client:
        Anything shaped like :class:`_ClientLike` — the real client from
        :meth:`from_settings`, or a stub in tests.
    model:
        The embedding model in force, reported by :attr:`model` so an
        index can record what wrote it. Changing the model invalidates an
        existing index (stated in ``.env.example``); this instance also
        learns the dimension from the first response and refuses a
        different one afterwards, so a mid-run change fails loudly
        instead of writing vectors two sizes into one store.
    batch_size:
        Texts per delegated request (validated: >= 1). Batches are
        validated one at a time, and the dimension learned from the
        first batch is enforced on every later one.
    retry_wait:
        Tenacity wait strategy between attempts (section 5.14 owns the
        policy). Injectable so tests exercise retry paths in
        milliseconds; production uses the exponential default.
    """

    def __init__(
        self,
        client: _ClientLike,
        *,
        model: str,
        batch_size: int = DEFAULT_BATCH_SIZE,
        retry_wait: WaitBaseT = DEFAULT_RETRY_WAIT,
    ) -> None:
        self._client = client
        self._model = model
        self._batch_size = require_batch_size(batch_size)
        self._retry_wait = retry_wait
        self._dimension: int | None = None

    @classmethod
    def from_settings(cls, settings: Settings) -> OpenAIEmbedder:
        """Build the real adapter from configuration.

        Raises ``SettingsError`` (a ``ConfigurationError``) through
        ``require_openai_key`` *before* any client exists, so a missing
        key fails with an actionable message instead of a cryptic 401
        from the API.
        """
        client = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=SecretStr(settings.require_openai_key()),
            # ``timeout`` is the constructor's alias for the
            # ``request_timeout`` field — pydantic names, both valid.
            timeout=settings.request_timeout_seconds,
            # Retries live in batching.py; with 0 here the transport
            # cannot stack a second, invisible policy on top of it.
            max_retries=0,
        )
        return cls(client, model=settings.embedding_model)

    @property
    def model(self) -> str:
        """The embedding model this instance writes with."""
        return self._model

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed every text in order — batched, retried, validated per batch.

        All-or-nothing: if a batch fails after exhausting retries the
        whole call raises, rather than returning a partial prefix the
        caller would have to reason about chunk by chunk.
        """
        validated = [require_embedding_text(text) for text in texts]
        if not validated:
            return []  # nothing to embed is not an error, and not a request
        vectors: list[list[float]] = []
        for batch in iter_batches(validated, self._batch_size):
            call = partial(self._client.embed_documents, batch)
            batch_vectors = self._call(call, what="embedding batch")
            vectors.extend(self._validate(batch_vectors, expected_count=len(batch)))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Embed one query into the same space, with the same checks."""
        require_embedding_text(text)
        vector = self._call(lambda: self._client.embed_query(text), what="embedding query")
        return self._validate([vector], expected_count=1)[0]

    def _call(self, operation: Callable[[], T], *, what: str) -> T:
        """Run one provider call through the retry policy, then translate.

        The retry wraps the *raw* client call so the policy can classify
        raw OpenAI exceptions precisely — 429/5xx/connection are retried,
        auth and 400 are not (section 5.14). Translation happens only after the
        policy has given up, so the error callers receive is the real
        final failure, just in our hierarchy.
        """
        try:
            return with_retries(operation, wait=self._retry_wait, what=what)
        except openai.OpenAIError as exc:
            raise _translate(exc) from exc

    # -- response validation ------------------------------------------------

    def _validate(self, vectors: object, *, expected_count: int) -> list[list[float]]:
        """Check a provider response before anything downstream sees it.

        Every rule here exists because of what a broken value would cost
        later: a short batch silently misaligns ``chunk_id`` *i* with
        vector *j* (citations would point at the wrong text); NaN poisons
        every cosine it touches; a dimension change would be rejected by
        the vector store much later, far from the cause.

        Raises ``DependencyError`` — the provider misbehaved, which is a
        502-class problem, not a caller mistake.
        """
        if not isinstance(vectors, list):
            raise DependencyError(
                "OpenAI returned a response that is not a list of embeddings.",
                code="embedding_invalid_response",
                detail=type(vectors).__name__,
            )
        if len(vectors) != expected_count:
            raise DependencyError(
                f"OpenAI returned {len(vectors)} embedding(s); expected {expected_count}.",
                code="embedding_count_mismatch",
                detail=f"model={self._model}",
            )
        dimension: int | None = None
        for position, vector in enumerate(vectors):
            if not isinstance(vector, list) or not vector:
                raise DependencyError(
                    f"Embedding {position} is not a non-empty list of numbers.",
                    code="embedding_invalid_vector",
                    detail=f"model={self._model}",
                )
            for component in vector:
                if not isinstance(component, float) or not math.isfinite(component):
                    raise DependencyError(
                        f"Embedding {position} contains a non-finite or non-float value.",
                        code="embedding_invalid_vector",
                        detail=f"model={self._model}",
                    )
            length = len(vector)
            if dimension is None:
                dimension = length
            elif length != dimension:
                raise DependencyError(
                    f"Embeddings in one batch disagree on dimension ({dimension} vs {length}).",
                    code="embedding_dimension_mismatch",
                    detail=f"model={self._model}",
                )
        # expected_count >= 1 here (empty batches return early), so the
        # loop ran and assigned the dimension the type checker wants.
        assert dimension is not None
        if self._dimension is None:
            self._dimension = dimension
        elif dimension != self._dimension:
            raise DependencyError(
                f"Embedding dimension changed from {self._dimension} to {dimension}; "
                "the model in force appears to have changed.",
                code="embedding_dimension_mismatch",
                detail=f"model={self._model}",
            )
        return vectors


def _translate(exc: openai.OpenAIError) -> RAGError:
    """Map one provider exception onto our hierarchy.

    The provider's own text is deliberately left behind (see the module
    docstring); ``detail`` carries only the exception class and HTTP
    status, and the original rides as ``__cause__``.
    """
    status = getattr(exc, "status_code", None)
    detail = f"openai.{type(exc).__name__}" + (f" HTTP {status}" if status else "")
    if isinstance(exc, openai.AuthenticationError):
        return ProviderAuthenticationError(
            "OpenAI rejected the configured API key.",
            detail=detail,
        )
    if isinstance(exc, openai.RateLimitError):
        return ProviderRateLimitError(
            "OpenAI is rate-limiting embedding requests; retry shortly.",
            detail=detail,
        )
    # Connection failures, timeouts, 5xx and any other status the
    # endpoint can produce: the dependency is unavailable, not our bug.
    return DependencyError(
        "The OpenAI embedding endpoint could not be reached or refused the request.",
        detail=detail,
    )
