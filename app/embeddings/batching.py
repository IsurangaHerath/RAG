"""Batched calls and bounded retries for embedding requests (Stage 21).

Embedding a document means one HTTP request per batch of texts, and the
network fails in two very different ways:

* **Too much at once.** Hundreds of chunk texts in one request are slow
  to build, awkward to retry, and hard to attribute when one value comes
  back wrong — so :func:`iter_batches` splits the work into consecutive,
  order-preserving slices: vector *i* stays paired with text *i*.
* **Not right now.** OpenAI answers 429 (rate limit) and 5xx (server
  trouble) with an implicit *try later* — so :func:`with_retries` tries
  again, with exponential backoff and jitter.

Retry only what can succeed later
---------------------------------
Rate limits, dropped connections and 5xx statuses are retried; every
other 4xx is not. A 401 means the key is wrong and a 400 means the
request is wrong — the identical retry makes the identical failure, so
those propagate immediately with the real error instead of buying
quadratic latency. Exceptions that are not OpenAI's at all (our own
programming bugs) are never retried either: waiting two seconds does not
fix a typo.

Bounded on purpose
------------------
Attempts are capped (``DEFAULT_RETRY_ATTEMPTS``), waits grow
exponentially but are capped too, and jitter keeps a burst of callers
from retrying in lockstep. Worst case adds seconds, not hours. On
exhaustion the *original* exception propagates (tenacity's ``reraise``)
so ``openai_embedder``'s translation still sees the true failure and
callers still get ``ProviderRateLimitError`` or ``DependencyError``.

One policy, visible
-------------------
The OpenAI client itself is built with ``max_retries=0`` (§5.13), so
this module is the only place that decides to wait. Two stacked retry
policies would multiply — three retries behind three retries means nine
requests — and nobody could reason about the worst case.

Tests inject ``attempts`` and ``wait`` to exercise exhaustion in
milliseconds instead of seconds; production uses the exponential
default. Nothing here logs document text or exception messages (which
can quote credentials) — only attempt counts, waits and type names.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence

import openai
from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)
from tenacity.wait import WaitBaseT

from app.domain.errors import ValidationError
from app.observability.logging import get_logger

__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_RETRY_ATTEMPTS",
    "DEFAULT_RETRY_WAIT",
    "WaitBaseT",
    "is_retryable_openai_error",
    "iter_batches",
    "require_batch_size",
    "with_retries",
]

#: Texts per request. §5.2's sketch suggested "e.g. 128"; large enough
#: that a 1000-chunk document costs a handful of requests, small enough
#: that a retried batch does not redo much work.
DEFAULT_BATCH_SIZE = 128

#: Total attempts including the first — one try plus three retries.
DEFAULT_RETRY_ATTEMPTS = 4

#: 1s, 2s, 4s, 8s... capped at ``max`` seconds, plus up to ``jitter``.
DEFAULT_RETRY_WAIT = wait_exponential_jitter(initial=1.0, max=8.0, jitter=0.5)

logger = get_logger(__name__)


def require_batch_size(batch_size: int) -> int:
    """Return ``batch_size`` if usable, else raise the contract's rejection.

    Shared by :func:`iter_batches` and the adapter's constructor so the
    rule is written once: a bad size fails on the line that set it, not
    as an empty batch somewhere mid-document.

    Raises
    ------
    ValidationError
        ``batch_size`` is below 1; code ``invalid_batch_size``.
    """
    if batch_size < 1:
        raise ValidationError(
            f"embedding batch size must be >= 1, got {batch_size}",
            code="invalid_batch_size",
        )
    return batch_size


def iter_batches[T](items: Sequence[T], batch_size: int) -> Iterator[list[T]]:
    """Yield ``items`` in consecutive batches of at most ``batch_size``.

    Order is preserved and nothing is dropped — downstream pairing of
    chunk ids and vectors depends on both. The final batch may be
    smaller; there is never an empty batch, and empty input yields
    nothing at all.
    """
    size = require_batch_size(batch_size)
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


def is_retryable_openai_error(exc: BaseException) -> bool:
    """Whether ``exc`` is worth trying again.

    Retryable:

    * ``APIConnectionError`` (and ``APITimeoutError`` beneath it) — the
      request may never have reached the provider, so nothing rejected it;
    * HTTP 429 and every 5xx — statuses that mean "later, maybe".

    Not retryable: any other 4xx (401/403: wrong key; 400: wrong
      request — the same attempt fails the same way), and anything that
      is not an OpenAI error at all, which would be our own bug wearing
      a network's clothes.
    """
    if isinstance(exc, openai.APIConnectionError):
        return True
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code == 429 or 500 <= exc.status_code < 600
    return False


def with_retries[T](
    operation: Callable[[], T],
    *,
    attempts: int = DEFAULT_RETRY_ATTEMPTS,
    wait: WaitBaseT = DEFAULT_RETRY_WAIT,
    what: str = "provider call",
) -> T:
    """Run ``operation``, retrying transient provider failures with backoff.

    Parameters
    ----------
    operation:
        A zero-argument callable making the provider request. Called
        again from the start on every attempt — each retry must be a
        complete, independent request.
    attempts:
        Total attempts including the first; must be >= 1.
    wait:
        Tenacity wait strategy between attempts. Injectable so tests can
        pass ``wait_none()`` and still exercise exhaustion.
    what:
        Short label naming the operation in retry log lines.

    Raises
    ------
    ValidationError
        ``attempts`` is below 1; code ``invalid_retry_attempts``.
    openai.OpenAIError
        The last retryable failure (after ``attempts``), or the first
        non-retryable one — always the original exception, never a
        wrapper, so the caller's translation sees the real cause.
    """
    if attempts < 1:
        raise ValidationError(
            f"retry attempts must be >= 1, got {attempts}",
            code="invalid_retry_attempts",
        )

    def _log_before_sleep(state: RetryCallState) -> None:
        _log_retry(state, what=what)

    retryer = Retrying(
        stop=stop_after_attempt(attempts),
        wait=wait,
        retry=retry_if_exception(is_retryable_openai_error),
        reraise=True,
        before_sleep=_log_before_sleep,
    )
    return retryer(operation)


def _log_retry(state: RetryCallState, *, what: str) -> None:
    """One warning per retry: attempt, wait, and the failure's *type*.

    The exception's message is deliberately left out: provider errors can
    quote the API key they rejected (§5.13), and no redaction rule
    should ever have to be trusted with a credential.
    """
    outcome = state.outcome
    error = outcome.exception() if outcome is not None else None
    next_action = state.next_action
    logger.warning(
        "embedding retry",
        operation=what,
        attempt=state.attempt_number,
        wait_seconds=round(next_action.sleep, 2) if next_action is not None else None,
        error=type(error).__name__ if error is not None else "unknown",
    )
