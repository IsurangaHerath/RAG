"""Tests for centralized configuration (Stage 06).

The two properties that matter most here:

1. **Validation** — impossible combinations (overlap >= chunk size, a
   nonsense log level) fail loudly at startup instead of causing a cryptic
   crash later during chunking.
2. **Secret safety** — the OpenAI key must never appear in any string a
   logger might emit.  These tests assert that on every serialization path
   pydantic offers, because it only takes one forgotten ``str()`` to leak
   a key into a log file.

All tests pass ``_env_file=None`` so a developer's real ``.env`` on disk
cannot change the results.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import Settings, SettingsError, get_settings

# A recognisable fake key. This is NOT a real credential.
FAKE_KEY = "sk-test-DO-NOT-USE-abc123SECRET"


def make_settings(**overrides: object) -> Settings:
    """Build a Settings object that ignores any real ``.env`` on disk."""
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


# --- Construction and defaults ------------------------------------------------


def test_defaults_are_valid() -> None:
    """Settings construct with no arguments (all defaults are safe)."""
    settings = make_settings()

    assert settings.chat_model == "gpt-4o-mini"
    assert settings.embedding_model == "text-embedding-3-large"
    assert settings.top_k == 5
    assert settings.chunk_size == 1000
    assert settings.chunk_overlap == 150


def test_env_file_is_not_required() -> None:
    """The app can be configured purely from real environment variables."""
    settings = Settings(_env_file=None, openai_api_key=FAKE_KEY)

    assert settings.has_openai_key is True


def test_environment_variable_overrides_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """A real environment variable wins over the field default."""
    monkeypatch.setenv("TOP_K", "17")
    monkeypatch.setenv("CHAT_MODEL", "gpt-4o")

    settings = Settings(_env_file=None)

    assert settings.top_k == 17, "env var TOP_K should override the default"
    assert settings.chat_model == "gpt-4o"


def test_unknown_environment_variables_are_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Irrelevant OS variables must not break startup."""
    monkeypatch.setenv("SOME_UNRELATED_SYSTEM_VAR", "anything")

    settings = Settings(_env_file=None)

    assert settings.top_k == 5


# --- Secret safety -------------------------------------------------------------


def test_api_key_is_recovered_by_require_openai_key() -> None:
    """The real key is available to code that genuinely needs it."""
    settings = make_settings(openai_api_key=FAKE_KEY)

    assert settings.require_openai_key() == FAKE_KEY


@pytest.mark.parametrize(
    "render",
    [
        repr,
        str,
        lambda s: s.model_dump_json(),
        lambda s: str(s.model_dump(mode="json")),
        lambda s: json.dumps(s.redacted()),
    ],
    ids=["repr", "str", "model_dump_json", "model_dump_json_mode", "redacted"],
)
def test_api_key_never_appears_in_serialized_form(render) -> None:
    """No common serialization path may expose the raw key.

    This is the central security test of Stage 06.
    """
    settings = make_settings(openai_api_key=FAKE_KEY)
    rendered = render(settings)

    assert FAKE_KEY not in rendered, f"API key leaked via {render}"


def test_redacted_masks_key_explicitly() -> None:
    """``redacted()`` is the intended logging payload and masks the key."""
    settings = make_settings(openai_api_key=FAKE_KEY)

    assert settings.redacted()["openai_api_key"] == "****"


def test_redacted_is_json_serialisable() -> None:
    """``redacted()`` can go straight into a JSON log line (paths become str)."""
    settings = make_settings(openai_api_key=FAKE_KEY)

    payload = json.dumps(settings.redacted())

    assert '"chat_model"' in payload
    assert FAKE_KEY not in payload


def test_missing_key_raises_helpful_error() -> None:
    """An unset key produces an actionable message, not an empty string."""
    settings = make_settings()

    assert settings.has_openai_key is False
    with pytest.raises(SettingsError, match="OPENAI_API_KEY is not set"):
        settings.require_openai_key()


def test_blank_key_treated_as_missing() -> None:
    """``OPENAI_API_KEY=`` in ``.env`` counts as 'not configured'."""
    settings = make_settings(openai_api_key="   ")

    assert settings.has_openai_key is False


# --- Validation ----------------------------------------------------------------


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    """Overlap >= chunk size would make the splitter loop forever."""
    with pytest.raises(ValidationError, match="must be smaller than"):
        make_settings(chunk_size=100, chunk_overlap=100)


def test_chunk_size_bounds_are_enforced() -> None:
    """Absurd chunk sizes are rejected at startup."""
    with pytest.raises(ValidationError):
        make_settings(chunk_size=5)  # below ge=50


@pytest.mark.parametrize("bad_level", ["LOUD", "verbose", "", "INFO2"])
def test_invalid_log_level_rejected(bad_level: str) -> None:
    """Only real logging levels are accepted."""
    with pytest.raises(ValidationError, match="log_level"):
        make_settings(log_level=bad_level)


@pytest.mark.parametrize("level", ["info", "INFO", "  warning  ", "debug"])
def test_log_level_normalised_to_uppercase(level: str) -> None:
    """Users may type any case; internally it is normalised."""
    settings = make_settings(log_level=level)

    assert settings.log_level == level.strip().upper()


def test_top_k_bounds_are_enforced() -> None:
    """top_k must be within its documented range."""
    with pytest.raises(ValidationError):
        make_settings(top_k=0)
    with pytest.raises(ValidationError):
        make_settings(top_k=1000)


def test_blank_model_name_rejected() -> None:
    """An empty model name would fail much later, at the API."""
    with pytest.raises(ValidationError, match="must not be empty"):
        make_settings(chat_model="   ")


def test_invalid_port_rejected() -> None:
    with pytest.raises(ValidationError):
        make_settings(api_port=99999)


def test_retrieval_threshold_none_is_allowed() -> None:
    """The default (no threshold) must remain valid."""
    settings = make_settings()

    assert settings.retrieval_threshold is None


def test_relative_paths_are_resolved_to_absolute(tmp_path: Path) -> None:
    """Paths do not depend on the current working directory."""
    settings = make_settings(data_dir=Path("some/relative/dir"))

    assert settings.data_dir.is_absolute()


# --- Directories -----------------------------------------------------------------


def test_ensure_directories_creates_all_paths(tmp_path: Path) -> None:
    """Data directories are created on demand."""
    settings = make_settings(
        data_dir=tmp_path / "data",
        chroma_persist_dir=tmp_path / "data" / "chroma",
        sample_docs_dir=tmp_path / "data" / "samples",
    )

    settings.ensure_directories()

    assert settings.data_dir.is_dir()
    assert settings.chroma_persist_dir.is_dir()
    assert settings.sample_docs_dir.is_dir()


def test_ensure_directories_is_idempotent(tmp_path: Path) -> None:
    """Calling it twice (e.g. on every request) must not raise."""
    settings = make_settings(
        data_dir=tmp_path / "data",
        chroma_persist_dir=tmp_path / "data" / "chroma",
        sample_docs_dir=tmp_path / "data" / "samples",
    )

    settings.ensure_directories()
    settings.ensure_directories()  # second call must be a no-op, not an error

    assert settings.data_dir.is_dir()


def test_chroma_dir_lives_under_data_dir_by_default() -> None:
    """Default layout keeps all local state inside the git-ignored data dir."""
    settings = make_settings(data_dir=Path("/somewhere/data").resolve())

    assert settings.chroma_persist_dir.parent == settings.data_dir


def test_overriding_data_dir_moves_its_subdirectories(tmp_path: Path) -> None:
    """Setting DATA_DIR alone must relocate ChromaDB and samples too.

    Regression test: originally the sub-directories stayed under the
    repository root, leaving state split across two places.
    """
    new_root = tmp_path / "relocated"
    settings = make_settings(data_dir=new_root)

    assert settings.data_dir == new_root.resolve()
    assert settings.chroma_persist_dir == (new_root / "chroma").resolve()
    assert settings.sample_docs_dir == (new_root / "samples").resolve()


def test_explicit_chroma_dir_overrides_the_derived_default(tmp_path: Path) -> None:
    """An explicit CHROMA_PERSIST_DIR still wins over data_dir."""
    custom = tmp_path / "somewhere_else"
    settings = make_settings(data_dir=tmp_path / "d", chroma_persist_dir=custom)

    assert settings.chroma_persist_dir == custom.resolve()
    assert settings.sample_docs_dir == (tmp_path / "d" / "samples").resolve()


# --- get_settings caching ----------------------------------------------------------


def test_get_settings_is_cached() -> None:
    """The settings object is built once per process, not per call."""
    get_settings.cache_clear()

    first = get_settings()
    second = get_settings()

    assert first is second, "get_settings() should return the same cached object"

    get_settings.cache_clear()  # leave the cache clean for other tests


# --- .env.example sanity ---------------------------------------------------------


def test_env_example_exists_and_has_no_real_secret() -> None:
    """``.env.example`` must exist and contain only a placeholder key."""
    root = Path(__file__).resolve().parents[2]
    example = root / ".env.example"

    assert example.is_file(), ".env.example is missing"

    content = example.read_text(encoding="utf-8")
    assert "OPENAI_API_KEY=" in content
    assert "REPLACE_WITH_YOUR_KEY" in content, (
        ".env.example should hold an obvious placeholder, not a real key"
    )

    # Heuristic: a real OpenAI key is 50+ chars starting sk- and has no dash
    # right after sk-.  Guards against a real key being pasted in by mistake.
    for line in content.splitlines():
        if line.startswith("OPENAI_API_KEY="):
            value = line.split("=", 1)[1].strip()
            assert value.startswith("sk-REPLACE"), (
                "Looks like a real key was committed to .env.example"
            )
            assert len(value) < 40, "Placeholder is suspiciously long — is it real?"


def test_env_file_is_gitignored() -> None:
    """A real ``.env`` must never be committable."""
    root = Path(__file__).resolve().parents[2]
    gitignore = (root / ".gitignore").read_text(encoding="utf-8")

    assert "\n.env\n" in f"\n{gitignore}", ".env should be ignored by git"
    assert ".env.example" in gitignore, ".env.example must remain committable"


def test_every_documented_env_example_key_is_a_settings_field() -> None:
    """No config option is documented in .env.example without existing.

    Prevents a user setting an environment variable that silently does
    nothing because it was never added to Settings.
    """
    root = Path(__file__).resolve().parents[2]
    content = (root / ".env.example").read_text(encoding="utf-8")

    field_names = set(Settings.model_fields)
    documented: set[str] = set()

    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key = line.split("=", 1)[0].strip()
        documented.add(key.lower())

    unknown = {
        key for key in documented if key.lower() not in {f.lower() for f in field_names}
    }
    assert not unknown, f".env.example documents unknown settings: {sorted(unknown)}"
