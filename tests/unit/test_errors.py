"""Tests for the shared error hierarchy (Stage 07).

The point of one hierarchy is that the API layer can map exceptions to HTTP
responses in a single place.  These tests pin the three things that mapping
depends on: a stable machine-readable ``code``, the correct ``http_status``,
and the guarantee that developer-only ``detail`` never reaches a client.
"""

from __future__ import annotations

import pytest

from app.config import Settings, SettingsError
from app.domain.errors import (
    ConfigurationError,
    DependencyError,
    NotFoundError,
    ProviderAuthenticationError,
    ProviderRateLimitError,
    RAGError,
    ValidationError,
)

# --- Structure -----------------------------------------------------------------


def test_all_errors_share_one_base() -> None:
    """A caller can catch RAGError once instead of a list of unrelated types."""
    assert issubclass(ConfigurationError, RAGError)
    assert issubclass(ValidationError, RAGError)
    assert issubclass(NotFoundError, RAGError)
    assert issubclass(DependencyError, RAGError)


def test_rag_error_is_a_runtime_error() -> None:
    """Backwards compatibility: SettingsError has raised RuntimeError since Stage 06."""
    assert issubclass(RAGError, RuntimeError)
    assert issubclass(SettingsError, RuntimeError)


def test_provider_errors_are_dependency_errors() -> None:
    """Rate limits and auth failures are both dependency problems (caught together)."""
    assert issubclass(ProviderAuthenticationError, DependencyError)
    assert issubclass(ProviderRateLimitError, DependencyError)


# --- Defaults and mapping --------------------------------------------------------


@pytest.mark.parametrize(
    ("error_cls", "expected_code", "expected_status"),
    [
        (RAGError, "internal_error", 500),
        (ConfigurationError, "configuration_error", 500),
        (ValidationError, "invalid_request", 400),
        (NotFoundError, "not_found", 404),
        (DependencyError, "dependency_unavailable", 502),
        (ProviderRateLimitError, "provider_rate_limited", 503),
    ],
    ids=lambda value: getattr(value, "__name__", str(value)),
)
def test_defaults(error_cls: type[RAGError], expected_code: str, expected_status: int) -> None:
    """Each error class carries a stable code and HTTP status."""
    err = error_cls("boom")

    assert err.code == expected_code
    assert err.http_status == expected_status
    assert err.message == "boom"


def test_code_and_status_can_be_overridden() -> None:
    """Callers may supply a more specific code without defining a new class."""
    err = ValidationError("bad top_k", code="top_k_out_of_range", http_status=422)

    assert err.code == "top_k_out_of_range"
    assert err.http_status == 422


def test_str_returns_message_only() -> None:
    """``str(err)`` stays readable in a log line (no metadata dumped)."""
    err = NotFoundError("doc_123 not found")

    assert str(err) == "doc_123 not found"


# --- Response payloads ------------------------------------------------------------


def test_to_dict_hides_detail_by_default() -> None:
    """Developer detail must not leak into an API response by accident."""
    err = DependencyError(
        "Search is temporarily unavailable",
        detail="chroma connect failed at D:\\private\\path",
    )

    payload = err.to_dict()

    assert payload == {
        "code": "dependency_unavailable",
        "message": "Search is temporarily unavailable",
    }
    assert "detail" not in payload


def test_to_dict_can_include_detail_when_asked() -> None:
    """Detail is available deliberately, e.g. when logging server-side."""
    err = DependencyError("Search is unavailable", detail="chroma timeout")

    assert err.to_dict(include_detail=True)["detail"] == "chroma timeout"


def test_to_dict_omits_empty_detail() -> None:
    """No ``detail: None`` noise in the payload."""
    payload = ValidationError("too long").to_dict(include_detail=True)

    assert "detail" not in payload


def test_detail_is_omitted_even_when_requested_if_absent() -> None:
    err = NotFoundError("gone")

    assert err.to_dict(include_detail=True) == {"code": "not_found", "message": "gone"}


# --- The errors actually raised today ----------------------------------------------


def test_settings_error_is_a_configuration_error() -> None:
    """SettingsError participates in the hierarchy, not off to one side."""
    err = SettingsError("OPENAI_API_KEY is not set")

    assert isinstance(err, ConfigurationError)
    assert isinstance(err, RAGError)
    assert err.code == "configuration_error"
    assert err.http_status == 500


def test_settings_error_from_real_settings_carries_http_mapping() -> None:
    """The path a real request would take: raise, then map to a response."""
    settings = Settings(_env_file=None)

    with pytest.raises(SettingsError) as excinfo:
        settings.require_openai_key()

    payload = excinfo.value.to_dict()
    assert payload["code"] == "configuration_error"
    assert excinfo.value.http_status == 500
    assert "OPENAI_API_KEY" in payload["message"]


def test_settings_construction_error_does_not_echo_a_secret() -> None:
    """A config error raised while a key IS set must still not leak it.

    Note: ``Settings(...)`` raises pydantic's ``ValidationError``, not
    ``RAGError``. That is intentional -- field validation belongs to pydantic,
    while ``RAGError`` is for failures this application raises itself.
    ``get_settings()`` converts the former into ``SettingsError`` at the
    boundary so callers see one type.
    """
    import pydantic

    secret = "sk-test-SHOULD-NEVER-APPEAR-xyz789"

    with pytest.raises(pydantic.ValidationError) as excinfo:
        # Force a validation failure on an object holding a real-looking key.
        Settings(_env_file=None, openai_api_key=secret, chunk_size=10)  # below ge=50

    rendered = str(excinfo.value) + repr(excinfo.value)
    assert secret not in rendered, "A validation error leaked the API key"


def test_get_settings_wraps_pydantic_errors_as_settings_error(monkeypatch) -> None:
    """The boundary converts pydantic errors so callers catch one type."""
    from app.config import get_settings

    # An invalid value in the environment makes Settings() fail to construct.
    monkeypatch.setenv("TOP_K", "not-a-number")
    get_settings.cache_clear()

    try:
        with pytest.raises(SettingsError) as excinfo:
            get_settings()

        # Callers see the unified type and its stable code.
        assert excinfo.value.code == "configuration_error"
        assert excinfo.value.http_status == 500
        assert isinstance(excinfo.value, ConfigurationError)
        # The wrapper is a RAGError, so the API layer maps it like any other.
        assert isinstance(excinfo.value, RAGError)
    finally:
        monkeypatch.delenv("TOP_K", raising=False)
        get_settings.cache_clear()  # leave the cache clean for other tests
