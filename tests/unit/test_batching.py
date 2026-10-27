"""Tests for batching and bounded retries (``app/embeddings/batching.py``).

Two properties carry the module: batches never lose or reorder texts,
and retries happen exactly when they should — every allowed attempt and
no more. Every test here runs in milliseconds because wait strategies
are injected (``wait_none()``) instead of slept through; exactly one
test elsewhere in the adapter suite exercises the real backoff.
"""

from __future__ import annotations

from types import SimpleNamespace

import openai
import pytest
from structlog.testing import capture_logs
from tenacity import wait_none

from app.domain.errors import ValidationError
from app.embeddings.batching import (
    DEFAULT_RETRY_ATTEMPTS,
    is_retryable_openai_error,
    iter_batches,
    require_batch_size,
    with_retries,
)
from tests.fakes import openai_connection_error, openai_status_error

# --- Batches: order preserved, nothing dropped ----------------------------


def test_batches_preserve_order_and_content() -> None:
    """Vector *i* must still pair with text *i* after splitting."""
    assert list(iter_batches(list("abcde"), 2)) == [list("ab"), list("cd"), list("e")]


def test_batch_larger_than_input_yields_a_single_batch() -> None:
    """A small document is one request, not several tiny ones."""
    assert list(iter_batches([1, 2, 3], 100)) == [[1, 2, 3]]


def test_empty_input_yields_no_batches() -> None:
    """Nothing to embed produces no requests at all."""
    assert list(iter_batches([], 4)) == []


@pytest.mark.parametrize("size", [0, -1])
def test_invalid_batch_size_is_rejected(size: int) -> None:
    """A bad size fails on the line that set it, not mid-document."""
    with pytest.raises(ValidationError) as excinfo:
        list(iter_batches(["a"], size))
    assert excinfo.value.code == "invalid_batch_size"


def test_require_batch_size_returns_the_value_when_valid() -> None:
    """The guard passes usable values straight through."""
    assert require_batch_size(128) == 128


# --- Retryability: what is worth trying again -----------------------------


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (openai_status_error(openai.RateLimitError, 429), True),
        (openai_status_error(openai.InternalServerError, 500), True),
        (openai_status_error(openai.APIStatusError, 503), True),
        (openai_connection_error(), True),
        (
            openai.APITimeoutError(request=SimpleNamespace(url="https://api.openai.com")),
            True,
        ),
        (openai_status_error(openai.AuthenticationError, 401), False),
        (openai_status_error(openai.BadRequestError, 400), False),
        (openai_status_error(openai.PermissionDeniedError, 403), False),
        (ValueError("our own bug"), False),
        (TypeError("also our own bug"), False),
    ],
)
def test_retryability_classification(error: BaseException, expected: bool) -> None:
    """Transient means later-maybe; auth/400/bugs mean never."""
    assert is_retryable_openai_error(error) is expected


# --- with_retries: bounded, faithful to the classification ----------------


def test_success_on_the_first_try_calls_once() -> None:
    """No failure, no second attempt, no wait."""
    calls: list[int] = []

    def measure() -> int:
        calls.append(1)
        return 42

    assert with_retries(measure, wait=wait_none()) == 42
    assert len(calls) == 1


def test_transient_failures_are_retried_until_success() -> None:
    """Two rate limits then success: the value comes back normally."""
    attempts = {"n": 0}

    def flaky() -> str:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise openai_status_error(openai.RateLimitError, 429)
        return "recovered"

    assert with_retries(flaky, attempts=4, wait=wait_none()) == "recovered"
    assert attempts["n"] == 3


def test_exhaustion_raises_the_original_error_after_all_attempts() -> None:
    """Bounded means bounded: exactly ``attempts`` calls, then the truth."""
    attempts = {"n": 0}

    def always_limited() -> None:
        attempts["n"] += 1
        raise openai_status_error(openai.RateLimitError, 429)

    with pytest.raises(openai.RateLimitError):
        with_retries(always_limited, attempts=3, wait=wait_none())
    assert attempts["n"] == 3


def test_auth_failures_are_not_retried() -> None:
    """A wrong key fails identically forever; return immediately."""
    attempts = {"n": 0}

    def rejected() -> None:
        attempts["n"] += 1
        raise openai_status_error(openai.AuthenticationError, 401)

    with pytest.raises(openai.AuthenticationError):
        with_retries(rejected, attempts=DEFAULT_RETRY_ATTEMPTS, wait=wait_none())
    assert attempts["n"] == 1


def test_our_own_bugs_are_not_retried() -> None:
    """Waiting two seconds does not fix a typo — no sleep, no hiding."""
    attempts = {"n": 0}

    def buggy() -> None:
        attempts["n"] += 1
        raise TypeError("bug in our code")

    with pytest.raises(TypeError):
        with_retries(buggy, attempts=DEFAULT_RETRY_ATTEMPTS, wait=wait_none())
    assert attempts["n"] == 1


def test_retry_attempts_must_be_positive() -> None:
    """Zero attempts is a configuration mistake, not a policy."""
    with pytest.raises(ValidationError) as excinfo:
        with_retries(lambda: 1, attempts=0, wait=wait_none())
    assert excinfo.value.code == "invalid_retry_attempts"


def test_retries_are_logged_without_the_exception_message() -> None:
    """Each retry logs attempt, wait and *type* — never provider text.

    Provider messages can quote the API key they rejected (§5.13), so
    the log line must carry the class name only.
    """
    secret = "sk-should-never-reach-a-log"
    attempts = {"n": 0}

    def limited() -> None:
        attempts["n"] += 1
        raise openai_status_error(openai.RateLimitError, 429, message=f"slow down for {secret}")

    with capture_logs() as captured:
        with pytest.raises(openai.RateLimitError):
            with_retries(
                limited,
                attempts=2,
                wait=wait_none(),
                what="embedding batch",
            )

    retries = [entry for entry in captured if entry["event"] == "embedding retry"]
    assert len(retries) == 1, captured
    entry = retries[0]
    assert entry["attempt"] == 1
    assert entry["operation"] == "embedding batch"
    assert entry["error"] == "RateLimitError"
    assert entry["wait_seconds"] == 0.0
    assert all(secret not in str(value) for value in entry.values())
    assert attempts["n"] == 2
