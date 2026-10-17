"""Consistent error handling (Stage 07).

Every module raises one of these exceptions instead of ad-hoc ``Exception``
or bare ``ValueError``.  The API layer later maps them to HTTP status codes
in exactly one place, so an error message never changes shape depending on
which route produced it.

Design rules for this hierarchy:

* A single :class:`RAGError` base so callers can catch everything at once.
* Each subclass carries an HTTP status code and a stable machine-readable
  ``code``, so clients can branch on ``code`` rather than parsing prose.
* ``detail`` is for developers/logs; ``message`` is safe to show a user.
  Never put an API key, absolute server path, or raw traceback in either.
"""

from __future__ import annotations

from typing import Any


class RAGError(RuntimeError):
    """Base class for every error this application raises deliberately.

    Derives from :class:`RuntimeError` so that ``SettingsError`` (which has
    raised ``RuntimeError`` since Stage 06) keeps its existing behaviour
    while now participating in this hierarchy.

    Parameters
    ----------
    message:
        Human-readable, safe to return to a client.
    code:
        Stable machine-readable identifier, e.g. ``"query_too_long"``.
    http_status:
        Status the API layer should respond with.
    detail:
        Optional developer context (logged, not returned to clients).
    """

    #: Subclasses override these to describe themselves.
    default_code: str = "internal_error"
    default_http_status: int = 500

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        http_status: int | None = None,
        detail: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code or self.default_code
        self.http_status = http_status or self.default_http_status
        self.detail = detail

    def to_dict(self, *, include_detail: bool = False) -> dict[str, Any]:
        """Return a JSON-serialisable payload for an API error response.

        ``detail`` is excluded by default because it may contain internals
        that should not leave the server.
        """
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if include_detail and self.detail:
            payload["detail"] = self.detail
        return payload

    def __str__(self) -> str:
        """Message only, so ``str(err)`` stays readable in logs."""
        return self.message


class ConfigurationError(RAGError):
    """Settings are missing or invalid. The process cannot run correctly."""

    default_code = "configuration_error"
    default_http_status = 500


class ValidationError(RAGError):
    """The caller supplied input that fails validation (400)."""

    default_code = "invalid_request"
    default_http_status = 400


class NotFoundError(RAGError):
    """A referenced document, chunk or resource does not exist (404)."""

    default_code = "not_found"
    default_http_status = 404


class DependencyError(RAGError):
    """An external dependency (OpenAI, ChromaDB) failed or is unreachable (502/503)."""

    default_code = "dependency_unavailable"
    default_http_status = 502


class DocumentLoadError(RAGError):
    """A file exists but its content cannot be read or decoded (422).

    Distinct from :class:`NotFoundError` (the file is there) and from
    :class:`ValidationError` (the caller's input was bad — here the *file*
    is bad). 422 rather than 400 because the request itself was well-formed;
    the payload simply cannot be processed.
    """

    default_code = "document_load_failed"
    default_http_status = 422


class ProviderAuthenticationError(DependencyError):
    """The upstream provider rejected our credentials (502, with a clear cause)."""

    default_code = "provider_auth_failed"


class ProviderRateLimitError(DependencyError):
    """The provider is rate-limiting us; callers may retry later (503)."""

    default_code = "provider_rate_limited"
    default_http_status = 503
