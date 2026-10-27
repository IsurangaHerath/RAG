"""Tests for the OpenAI adapter: validation, batching, retries, wiring.

Everything here runs offline. The adapter's delegate is a stub shaped
like ``OpenAIEmbeddings``, and every adapter is built with an instant
retry wait, so what is tested is *our* behaviour — input rules, batch
splitting, response checks, and how provider failures become the
project's error hierarchy — never OpenAI's, and never a real sleep.
The contract suite in ``test_embedder_contract.py`` runs the adapter
through the same shared rules as every other implementation; the retry
policy itself is tested in ``test_batching.py``.
"""

from __future__ import annotations

import openai
import pytest
from langchain_openai import OpenAIEmbeddings
from tenacity import wait_none

from app.config import Settings, SettingsError
from app.domain.errors import (
    DependencyError,
    ProviderAuthenticationError,
    ProviderRateLimitError,
    ValidationError,
)
from app.embeddings.openai_embedder import OpenAIEmbedder
from tests.fakes import (
    FakeEmbedder,
    StubEmbeddings,
    openai_connection_error,
    openai_status_error,
)

MODEL = "text-embedding-3-large"


def _adapter(stub: StubEmbeddings, **kwargs: object) -> OpenAIEmbedder:
    """Build the adapter under test with instant retries (no sleeping)."""
    return OpenAIEmbedder(stub, model=MODEL, retry_wait=wait_none(), **kwargs)


# --- Input rules: nothing reaches the provider ----------------------------


def test_blank_input_never_reaches_the_delegate() -> None:
    """Validation happens locally: no call, no cost, stable error code."""
    stub = StubEmbeddings()
    adapter = _adapter(stub)
    with pytest.raises(ValidationError):
        adapter.embed_documents(["", "   "])
    with pytest.raises(ValidationError):
        adapter.embed_query("")
    assert stub.calls == 0


def test_empty_batch_short_circuits_without_a_request() -> None:
    """Nothing to embed is an empty result, not a wasted round trip."""
    stub = StubEmbeddings()
    assert _adapter(stub).embed_documents([]) == []
    assert stub.calls == 0


# --- Delegation: values survive the round trip unchanged ------------------


def test_documents_are_passed_through_in_order() -> None:
    """What the stub computes is what the adapter returns — no reordering."""
    texts = ["Rate limits apply per account.", "Deploy with docker compose up -d."]
    assert _adapter(StubEmbeddings()).embed_documents(texts) == FakeEmbedder().embed_documents(
        texts
    )


def test_queries_use_the_clients_query_path() -> None:
    """Queries go through ``embed_query``, not through a one-element batch."""
    stub = StubEmbeddings()
    _adapter(stub).embed_query("how do rate limits work?")
    assert stub.queries == ["how do rate limits work?"]


# --- Batching: split in order, reassemble in order ------------------------


def test_batches_are_split_in_order_and_results_concatenated() -> None:
    """Five texts at batch size 2 means three requests, same final order."""
    stub = StubEmbeddings()
    adapter = _adapter(stub, batch_size=2)
    texts = ["alpha", "beta", "gamma", "delta", "epsilon"]
    assert adapter.embed_documents(texts) == FakeEmbedder().embed_documents(texts)
    assert stub.batches == [["alpha", "beta"], ["gamma", "delta"], ["epsilon"]]


def test_constructor_rejects_an_invalid_batch_size() -> None:
    """A bad size fails on the line that set it, not mid-document."""
    with pytest.raises(ValidationError) as excinfo:
        _adapter(StubEmbeddings(), batch_size=0)
    assert excinfo.value.code == "invalid_batch_size"


def test_transient_rate_limit_is_retried_until_success() -> None:
    """One 429 from the provider, then normal service: one extra call."""
    stub = StubEmbeddings()
    stub.errors = [openai_status_error(openai.RateLimitError, 429)]
    adapter = _adapter(stub)
    vectors = adapter.embed_documents(["hello world"])
    assert len(vectors) == 1
    assert stub.calls == 2


# --- Response validation: the provider does not get the benefit of doubt --


def test_wrong_vector_count_is_rejected() -> None:
    """A short batch would misalign chunk ids with vectors downstream."""
    stub = StubEmbeddings()
    stub.mutate = lambda vectors: vectors[:-1]
    with pytest.raises(DependencyError) as excinfo:
        _adapter(stub).embed_documents(["alpha", "beta"])
    assert excinfo.value.code == "embedding_count_mismatch"


def test_response_that_is_not_a_list_is_rejected() -> None:
    """A structurally impossible response is a dependency failure."""
    stub = StubEmbeddings()

    def _garbage(_vectors: list[list[float]]) -> list[list[float]]:
        return "surprise"  # type: ignore[return-value]

    stub.mutate = _garbage
    with pytest.raises(DependencyError) as excinfo:
        _adapter(stub).embed_documents(["alpha"])
    assert excinfo.value.code == "embedding_invalid_response"


def test_empty_vector_is_rejected() -> None:
    """A zero-length vector has no direction to compare."""
    stub = StubEmbeddings()
    stub.mutate = lambda vectors: [[] for _ in vectors]
    with pytest.raises(DependencyError) as excinfo:
        _adapter(stub).embed_documents(["alpha"])
    assert excinfo.value.code == "embedding_invalid_vector"


def test_non_float_component_is_rejected() -> None:
    """Integers arrive from JSON-less paths; floats only, or the store says no."""
    stub = StubEmbeddings()
    stub.mutate = lambda vectors: [[0 for _ in vector] for vector in vectors]
    with pytest.raises(DependencyError) as excinfo:
        _adapter(stub).embed_documents(["alpha"])
    assert excinfo.value.code == "embedding_invalid_vector"


def test_non_finite_component_is_rejected() -> None:
    """One NaN would poison every cosine the vector takes part in."""
    stub = StubEmbeddings()
    stub.mutate = lambda vectors: [
        [float("nan") if index == 0 else value for index, value in enumerate(vector)]
        for vector in vectors
    ]
    with pytest.raises(DependencyError) as excinfo:
        _adapter(stub).embed_documents(["alpha"])
    assert excinfo.value.code == "embedding_invalid_vector"


def test_ragged_batch_is_rejected() -> None:
    """Vectors in one batch must agree on dimension — the store requires it."""
    stub = StubEmbeddings()
    stub.mutate = lambda vectors: [
        vector[:8] if position == 0 else vector for position, vector in enumerate(vectors)
    ]
    with pytest.raises(DependencyError) as excinfo:
        _adapter(stub).embed_documents(["alpha", "beta"])
    assert excinfo.value.code == "embedding_dimension_mismatch"


def test_dimension_change_across_calls_is_rejected() -> None:
    """A model changing mid-run must fail loudly, not write mixed sizes."""
    stub = StubEmbeddings()
    stub.mutate = lambda vectors: [vector[:8] for vector in vectors]
    adapter = _adapter(stub)
    adapter.embed_documents(["first"])  # the instance learns dimension 8
    stub.mutate = None
    with pytest.raises(DependencyError) as excinfo:
        adapter.embed_documents(["second"])
    assert excinfo.value.code == "embedding_dimension_mismatch"
    assert "changed from 8 to 64" in str(excinfo.value)


def test_query_responses_run_through_the_same_validation() -> None:
    """A query's single vector gets the checks a batch gets."""
    stub = StubEmbeddings()
    stub.mutate = lambda vectors: [[] for _ in vectors]
    with pytest.raises(DependencyError) as excinfo:
        _adapter(stub).embed_query("hello")
    assert excinfo.value.code == "embedding_invalid_vector"


# --- Error translation: provider failures join the RAGError hierarchy ----


@pytest.mark.parametrize(
    ("provider_cls", "status", "expected_type", "expected_code"),
    [
        (
            openai.AuthenticationError,
            401,
            ProviderAuthenticationError,
            "provider_auth_failed",
        ),
        (
            openai.RateLimitError,
            429,
            ProviderRateLimitError,
            "provider_rate_limited",
        ),
        (openai.InternalServerError, 500, DependencyError, "dependency_unavailable"),
        (openai.BadRequestError, 400, DependencyError, "dependency_unavailable"),
    ],
)
def test_provider_status_errors_are_translated(
    provider_cls: type[openai.APIStatusError],
    status: int,
    expected_type: type[DependencyError],
    expected_code: str,
) -> None:
    """Each provider failure lands on the matching project error.

    For retryable statuses (429, 500) this also proves translation
    happens *after* the retries give up, not instead of them.
    """
    stub = StubEmbeddings()
    stub.error = openai_status_error(provider_cls, status)
    with pytest.raises(expected_type) as excinfo:
        _adapter(stub).embed_documents(["hello world"])
    error = excinfo.value
    assert error.code == expected_code
    assert provider_cls.__name__ in (error.detail or "")
    assert isinstance(error.__cause__, openai.OpenAIError)


def test_connection_failures_become_dependency_errors() -> None:
    """Unreachable endpoint is a 502-class dependency problem."""
    stub = StubEmbeddings()
    stub.error = openai_connection_error()
    with pytest.raises(DependencyError) as excinfo:
        _adapter(stub).embed_query("hello")
    error = excinfo.value
    assert error.code == "dependency_unavailable"
    assert "APIConnectionError" in (error.detail or "")


def test_provider_message_never_leaks_into_our_error() -> None:
    """OpenAI's auth errors quote the rejected key; ours must not.

    ``message`` and ``detail`` travel into logs and API responses, so
    only the exception class and HTTP status survive the translation.
    """
    secret = "sk-live-ABC123secret"
    stub = StubEmbeddings()
    stub.error = openai_status_error(
        openai.AuthenticationError,
        401,
        message=f"Incorrect API key provided: {secret}",
    )
    with pytest.raises(ProviderAuthenticationError) as excinfo:
        _adapter(stub).embed_query("hello")
    error = excinfo.value
    assert secret not in str(error)
    assert secret not in (error.detail or "")
    assert error.detail == "openai.AuthenticationError HTTP 401"


# --- Construction from settings -------------------------------------------


def test_from_settings_wires_configuration_into_the_client() -> None:
    """Model, timeout and retry policy land on the real langchain client."""
    settings = Settings(
        _env_file=None,
        openai_api_key="sk-test-not-a-real-key",
        request_timeout_seconds=12.5,
    )
    adapter = OpenAIEmbedder.from_settings(settings)
    assert adapter.model == settings.embedding_model
    client = adapter._client
    assert isinstance(client, OpenAIEmbeddings)
    assert client.model == settings.embedding_model
    assert client.request_timeout == 12.5
    # One retry policy, and it lives in batching.py — the transport must
    # not stack a second, invisible one on top.
    assert client.max_retries == 0


def test_from_settings_without_a_key_fails_before_building_anything(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No key means a helpful SettingsError, not a cryptic 401 later."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = Settings(_env_file=None)
    with pytest.raises(SettingsError):
        OpenAIEmbedder.from_settings(settings)
