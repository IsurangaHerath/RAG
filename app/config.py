"""Centralized, validated application configuration (Stage 06).

All tunables live in one typed :class:`Settings` object so no other module
has to read ``os.environ`` directly.  Values come from (in priority order):

1. Real environment variables (``OPENAI_API_KEY=... python app.py``)
2. A local ``.env`` file (git-ignored; copy ``.env.example`` to get started)
3. The field defaults defined below

Why this matters for a beginner: hardcoding an API key "just for now" is
the single most common way secrets end up on GitHub.  Here the key is a
:class:`pydantic.SecretStr`, which pydantic masks automatically, so even an
accidental ``print(settings)`` or ``logger.info(settings)`` cannot leak it.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root: this file is <root>/app/config.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SettingsError(RuntimeError):
    """Raised when configuration is present but invalid or unusable."""


class Settings(BaseSettings):
    """Typed application settings loaded from environment variables or ``.env``.

    Attributes are grouped by concern.  Every value has a safe default so
    that ``get_settings()`` works out of the box for local development,
    with one exception: ``openai_api_key`` must be provided before any
    live OpenAI call is attempted.
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",  # ignore unrelated OS variables instead of failing
        case_sensitive=False,
        validate_default=True,
    )

    # --- LLM / provider ---------------------------------------------------
    openai_api_key: SecretStr | None = Field(
        default=None,
        description="OpenAI API key. SecretStr so it is never printed or logged.",
    )
    chat_model: str = Field(
        default="gpt-4o-mini",
        description="Chat completion model used for grounded answers.",
    )
    embedding_model: str = Field(
        default="text-embedding-3-large",
        description="Embedding model. Changing it invalidates the ChromaDB index.",
    )
    request_timeout_seconds: float = Field(
        default=60.0,
        ge=1,
        le=600,
        description="Per-request timeout for provider calls.",
    )
    max_output_tokens: int = Field(
        default=1024,
        ge=1,
        le=32768,
        description="Upper bound on tokens the LLM may generate.",
    )

    # --- Data directories --------------------------------------------------
    data_dir: Path = Field(
        default=PROJECT_ROOT / "data",
        description="Root for uploads, sample documents and generated data.",
    )
    chroma_persist_dir: Path = Field(
        default=PROJECT_ROOT / "data" / "chroma",
        description="ChromaDB persistent storage directory.",
    )
    sample_docs_dir: Path = Field(
        default=PROJECT_ROOT / "data" / "samples",
        description="Directory of sample technical documents (Stage 17).",
    )

    # --- Chunking -----------------------------------------------------------
    chunk_size: int = Field(
        default=1000,
        ge=50,
        le=8000,
        description="Maximum characters per chunk.",
    )
    chunk_overlap: int = Field(
        default=150,
        ge=0,
        le=4000,
        description="Characters shared between neighbouring chunks.",
    )

    # --- Retrieval -----------------------------------------------------------
    top_k: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Number of chunks returned by a query.",
    )
    retrieval_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional score cutoff. Requires calibration; None disables it.",
    )

    # --- HTTP limits ---------------------------------------------------------
    max_upload_mb: int = Field(
        default=20,
        ge=1,
        le=500,
        description="Maximum accepted upload size in megabytes.",
    )
    max_query_chars: int = Field(
        default=4000,
        ge=1,
        le=100_000,
        description="Maximum accepted question length.",
    )
    api_host: str = Field(default="127.0.0.1", description="Bind address for uvicorn.")
    api_port: int = Field(default=8000, ge=1, le=65535, description="Port for uvicorn.")

    # --- Logging ---------------------------------------------------------------
    log_level: str = Field(
        default="INFO",
        description="Python logging level.",
    )
    log_document_content: bool = Field(
        default=False,
        description=(
            "When False (the default) document text is never written to logs. "
            "Only enable for local debugging."
        ),
    )

    # --- Validation ----------------------------------------------------------

    @field_validator("log_level")
    @classmethod
    def _normalise_log_level(cls, value: str) -> str:
        """Accept ``info`` or ``INFO`` but reject nonsense like ``LOUD``."""
        level = value.strip().upper()
        allowed = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}
        if level not in allowed:
            raise ValueError(
                f"log_level must be one of {sorted(allowed)}; got {value!r}"
            )
        return level

    @field_validator("chat_model", "embedding_model")
    @classmethod
    def _reject_blank_model(cls, value: str) -> str:
        """A model name must be a non-empty string."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("model name must not be empty")
        return cleaned

    @field_validator("data_dir", "chroma_persist_dir", "sample_docs_dir")
    @classmethod
    def _absolute_paths(cls, value: Path) -> Path:
        """Resolve relative paths so behaviour does not depend on the CWD."""
        return value.expanduser().resolve()

    @model_validator(mode="after")
    def _derive_data_paths(self) -> Settings:
        """Keep sub-directories inside ``data_dir`` unless explicitly overridden.

        Without this, setting ``DATA_DIR=/mnt/big`` would move uploads but
        leave ChromaDB indexing under the repository root -- surprising, and
        exactly the kind of split state that is hard to clean up later.
        ``model_fields_set`` tells us which values the caller actually
        supplied, so an explicit ``CHROMA_PERSIST_DIR`` is still honoured.
        """
        if "chroma_persist_dir" not in self.model_fields_set:
            self.chroma_persist_dir = (self.data_dir / "chroma").resolve()
        if "sample_docs_dir" not in self.model_fields_set:
            self.sample_docs_dir = (self.data_dir / "samples").resolve()
        return self

    @model_validator(mode="after")
    def _check_overlap(self) -> Settings:
        """Overlap must be smaller than the chunk size, or splitting loops."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be smaller than "
                f"chunk_size ({self.chunk_size})"
            )
        return self

    # --- Derived helpers -------------------------------------------------------

    @property
    def has_openai_key(self) -> bool:
        """True when an OpenAI API key has been configured."""
        return self.openai_api_key is not None and bool(
            self.openai_api_key.get_secret_value().strip()
        )

    def require_openai_key(self) -> str:
        """Return the API key, or raise a helpful error if it is missing.

        We deliberately fail here rather than sending an empty key to the
        API, so the user gets an actionable message instead of a 401.
        """
        if not self.has_openai_key:
            raise SettingsError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and add "
                "your key, or export the OPENAI_API_KEY environment variable."
            )
        assert self.openai_api_key is not None  # for type checkers
        return self.openai_api_key.get_secret_value()

    def ensure_directories(self) -> None:
        """Create the data directories if they do not already exist.

        Called once at startup; it is idempotent, so repeated calls are safe.
        """
        for directory in (self.data_dir, self.chroma_persist_dir, self.sample_docs_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def redacted(self) -> dict[str, object]:
        """Return a dict that is safe to log — the API key is masked.

        Use this instead of ``settings.model_dump()`` anywhere near a logger.
        """
        data: dict[str, object] = self.model_dump()
        if self.openai_api_key is not None:
            data["openai_api_key"] = "****"
        # Paths become strings so they serialise cleanly to JSON logs.
        for key, value in list(data.items()):
            if isinstance(value, Path):
                data[key] = str(value)
        return data


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide :class:`Settings` instance.

    Cached because pydantic re-reads and re-validates the ``.env`` file on
    every instantiation, and settings are immutable for a process's lifetime.

    Tests must call ``get_settings.cache_clear()`` after changing env vars.
    """
    try:
        return Settings()
    except Exception as exc:  # pragma: no cover - exercised via explicit tests
        raise SettingsError(f"Invalid configuration: {exc}") from exc
