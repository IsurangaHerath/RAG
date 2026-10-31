"""Tests for structured logging and its redaction guarantee.

These tests are mostly adversarial on purpose. The module's contract is not
"document text is usually omitted" — it is that a careless call cannot put a
document's contents into a log store that outlives the index it was
debugging. So the suite attacks it from the directions a real leak happens:
a flagged key, an innocently-named key with a large value, a container
holding text, and the one nobody thinks about — an *object* whose ``repr``
quotes its own content.

``structlog`` and the root logger are process-global, so every test here
runs under a fixture that restores both; without it one test's
reconfiguration would silently define the next one's behaviour.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from enum import Enum
from io import StringIO
from pathlib import Path

import pytest
import structlog

from app.config import Settings
from app.domain.models import Document
from app.observability.logging import (
    CONTENT_KEYS,
    DEFAULT_MAX_VALUE_CHARS,
    REDACTED,
    configure_from_settings,
    configure_logging,
    get_logger,
    make_content_redactor,
)

SECRET = "confidential clause nine must not reach a log"


def redact(
    payload: dict[str, object],
    *,
    allow_document_content: bool = False,
    max_value_chars: int = DEFAULT_MAX_VALUE_CHARS,
) -> dict[str, object]:
    """Run one payload through the processor exactly as structlog would."""
    processor = make_content_redactor(
        allow_document_content=allow_document_content,
        max_value_chars=max_value_chars,
    )
    return dict(processor(None, "info", dict(payload)))  # type: ignore[operator]


@pytest.fixture(autouse=True)
def restore_logging() -> object:
    """Put structlog and the root logger back after every test."""
    saved_config = structlog.get_config()
    root = logging.getLogger()
    saved_level = root.level
    saved_handlers = list(root.handlers)
    yield
    structlog.configure(**saved_config)  # type: ignore[arg-type]
    root.setLevel(saved_level)
    root.handlers[:] = saved_handlers


def capture(**kwargs: object) -> tuple[StringIO, object]:
    """Configure logging and return a stream receiving the rendered output."""
    configure_logging(**kwargs)  # type: ignore[arg-type]
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger().handlers = [handler]
    return stream, get_logger("test.probe")


def configure_from_settings_and_capture(settings: Settings) -> tuple[StringIO, object]:
    """Apply settings and hand back a stream plus a logger writing to it."""
    configure_from_settings(settings)
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger().handlers = [handler]
    return stream, get_logger("test.probe")


# --- the key-name deny-list --------------------------------------------------


def test_flagged_key_is_redacted_even_when_short() -> None:
    """A one-line secret is still a secret; length does not excuse it."""
    out = redact({"event": "loaded", "content": SECRET})

    assert out["content"] == REDACTED
    assert SECRET not in json.dumps(out)


def test_key_matching_is_case_insensitive() -> None:
    assert redact({"Content": SECRET})["Content"] == REDACTED
    assert redact({"PAGE_CONTENT": SECRET})["PAGE_CONTENT"] == REDACTED


def test_every_declared_key_is_lowercase() -> None:
    """The processor lowercases before matching, so the set must agree."""
    assert all(key == key.lower() for key in CONTENT_KEYS)


def test_ordinary_short_string_survives() -> None:
    assert redact({"event": "a short message"})["event"] == "a short message"


def test_numbers_booleans_and_none_survive() -> None:
    out = redact({"event": "x", "loaded": 8, "ratio": 0.5, "ok": True, "gone": None})

    assert out["loaded"] == 8
    assert out["ratio"] == 0.5
    assert out["ok"] is True
    assert out["gone"] is None


def test_a_count_containing_the_word_text_is_not_redacted() -> None:
    """The reason the deny-list is exact-match, not a substring search.

    ``"text" in key`` would swallow this count along with a document body,
    and a log that lies about how many text files it saw is worse than one
    that shows too little.
    """
    out = redact({"event": "x", "text_file_count": 3})

    assert out["text_file_count"] == 3


# --- the size guard ----------------------------------------------------------


def test_long_value_under_an_innocent_key_is_withheld() -> None:
    """The backstop for key names the deny-list has never heard of."""
    out = redact({"event": "loaded", "note": "A" * 600})

    assert out["note"] == "<omitted 600 chars>"
    assert "A" * 600 not in json.dumps(out)


def test_value_exactly_at_the_threshold_survives() -> None:
    """The boundary is inclusive: length > max is what gets withheld."""
    out = redact({"event": "x", "note": "B" * 10}, max_value_chars=10)

    assert out["note"] == "B" * 10


def test_value_one_over_the_threshold_is_withheld() -> None:
    out = redact({"event": "x", "note": "B" * 11}, max_value_chars=10)

    assert out["note"] == "<omitted 11 chars>"


def test_default_threshold_sits_below_the_chunk_size() -> None:
    """A whole chunk must never fit through the guard.

    Chunk size defaults to 1000 characters (section 7.0), so a default below that
    guarantees a chunk cannot arrive intact.
    """
    assert DEFAULT_MAX_VALUE_CHARS < 1000

    out = redact({"event": "x", "note": SECRET + " " + "word " * 300})

    assert out["note"].startswith("<omitted ")


def test_zero_threshold_withholds_every_non_empty_string() -> None:
    out = redact({"event": "x", "empty": "", "one": "a"}, max_value_chars=0)

    assert out["empty"] == ""
    assert out["one"] == "<omitted 1 chars>"


def test_negative_threshold_is_rejected() -> None:
    with pytest.raises(ValueError, match="max_value_chars"):
        make_content_redactor(max_value_chars=-1)


# --- containers --------------------------------------------------------------


def test_flagged_key_inside_a_dict_is_redacted() -> None:
    out = redact({"event": "x", "payload": {"content": SECRET, "kept": 1}})

    assert out["payload"] == {"content": REDACTED, "kept": 1}  # type: ignore[assignment]


def test_a_flagged_key_is_blanked_wholesale_including_its_structure() -> None:
    """Redaction at a flagged key wins over descending into its value.

    ``chunks`` says the value *is* content, so the whole value goes —
    including any ids nested beside the text. The alternative, walking in
    and redacting leaf strings one shape at a time, stays safe only for as
    long as every future container shape is handled correctly; blanking at
    the key the caller wrote is the rule that cannot be misused.
    """
    out = redact({"event": "x", "chunks": [{"id": 7, "text": SECRET}]})

    assert out["chunks"] == REDACTED
    assert SECRET not in json.dumps(out)


def test_a_list_under_a_flagged_key_is_blanked_wholesale() -> None:
    out = redact({"event": "x", "chunks": [SECRET, SECRET]})

    assert out["chunks"] == REDACTED


def test_long_strings_are_withheld_at_depth() -> None:
    out = redact({"event": "x", "batch": {"notes": ["C" * 600]}})

    assert out["batch"] == {"notes": ["<omitted 600 chars>"]}  # type: ignore[assignment]


def test_very_deep_nesting_is_collapsed_not_walked_forever() -> None:
    deep: dict[str, object] = {"event": "x"}
    cursor: dict[str, object] = deep
    for index in range(10):
        cursor["level"] = {"next": index}
        cursor = cursor["level"]  # type: ignore[assignment]

    out = redact(deep)

    assert "nested too deeply" in json.dumps(out)


# --- arbitrary objects -------------------------------------------------------


def test_a_document_object_is_reduced_to_its_type_name() -> None:
    """The leak the other two layers cannot see.

    ``Document.__repr__`` quotes its own ``content``, so handing one to a
    logger would smuggle the whole text past both a key-name check and a
    length check — the value is neither a flagged key nor a long string.
    """
    doc = Document.create(
        source_path="a.md",
        filename="a.md",
        media_type="text/markdown",
        content=SECRET,
    )

    out = redact({"event": "x", "record": doc})

    assert out["record"] == "<Document>"
    assert SECRET not in json.dumps(out)


def test_a_path_becomes_a_plain_string() -> None:
    """Common in logs, harmless, and nicer than the renderer's repr()."""
    out = redact({"event": "x", "root": Path("data") / "docs"})

    assert out["root"] == str(Path("data") / "docs")


def test_an_enum_becomes_its_value() -> None:
    class Colour(Enum):
        RED = "red"

    assert redact({"event": "x", "colour": Colour.RED})["colour"] == "red"


def test_a_datetime_becomes_an_iso_string() -> None:
    moment = datetime(2026, 10, 22, tzinfo=UTC)

    assert redact({"event": "x", "at": moment})["at"] == moment.isoformat()


def test_bytes_are_withheld_with_their_length() -> None:
    out = redact({"event": "x", "blob": b"\x00" * 10})

    assert out["blob"] == "<omitted 10 bytes>"


# --- the escape hatch --------------------------------------------------------


def test_allow_flag_passes_document_text_through() -> None:
    out = redact({"event": "loaded", "content": SECRET}, allow_document_content=True)

    assert out["content"] == SECRET


def test_allow_flag_also_disables_the_size_guard() -> None:
    out = redact({"event": "x", "note": "D" * 600}, allow_document_content=True)

    assert out["note"] == "D" * 600


def test_allow_flag_leaves_the_payload_keys_intact() -> None:
    payload = {"event": "x", "content": SECRET, "loaded": 1}

    assert set(redact(payload, allow_document_content=True)) == set(payload)


# --- configure_logging -------------------------------------------------------


def test_configured_logger_emits_parseable_json() -> None:
    stream, log = capture()

    log.info("file loaded", source="a.md")

    record = json.loads(stream.getvalue().strip())
    assert record["event"] == "file loaded"
    assert record["source"] == "a.md"


def test_configured_output_contains_no_document_content() -> None:
    stream, log = capture()

    log.info("file loaded", source="a.md", content=SECRET)

    output = stream.getvalue()
    assert SECRET not in output
    assert REDACTED in output


def test_configured_output_withholds_a_long_value() -> None:
    stream, log = capture()

    log.info("dump", note="E" * 600)

    assert "<omitted 600 chars>" in stream.getvalue()


def test_configured_output_reduces_a_document_to_its_name() -> None:
    stream, log = capture()
    doc = Document.create(
        source_path="a.md",
        filename="a.md",
        media_type="text/markdown",
        content=SECRET,
    )

    log.info("record", record=doc)

    output = stream.getvalue()
    assert "<Document>" in output
    assert SECRET not in output


def test_the_log_records_the_level() -> None:
    stream, log = capture()

    log.warning("careful now")

    assert json.loads(stream.getvalue())["level"] == "warning"


def test_console_renderer_still_redacts() -> None:
    """A readable format is not a licence to be careless."""
    stream, log = capture(renderer="console")

    log.info("file loaded", content=SECRET, note="F" * 600)

    output = stream.getvalue()
    assert SECRET not in output
    assert "<omitted 600 chars>" in output


def test_an_unknown_level_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown log level"):
        configure_logging(level="LOUD")


def test_level_is_case_and_whitespace_tolerant() -> None:
    """Same rule Settings applies to LOG_LEVEL."""
    stream, log = capture(level="  debug  ")

    log.debug("detailed")

    assert json.loads(stream.getvalue())["level"] == "debug"


def test_rerunning_configuration_applies_the_new_processors() -> None:
    """Safe to call twice — and the second call must actually win."""
    stream, _ = capture()
    stream.truncate(0)
    stream.seek(0)

    configure_logging(allow_document_content=True)
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger().handlers = [handler]
    log = get_logger("test.probe")

    log.info("file loaded", content=SECRET)

    assert SECRET in stream.getvalue()


def test_a_process_starting_with_no_handlers_gets_one() -> None:
    """Otherwise records routed through the stdlib vanish in silence.

    Under pytest the root logger always already has a capture handler, so
    this is the branch a freshly-started process takes — the one where
    nothing would be written at all without the fallback.
    """
    root = logging.getLogger()
    root.handlers = []

    configure_logging(level="INFO")

    assert len(root.handlers) == 1
    assert isinstance(root.handlers[0], logging.StreamHandler)
    assert root.handlers[0].formatter is not None


def test_settings_defaults_keep_redaction_on() -> None:
    """A fresh install gets the safe behaviour without asking for it."""
    assert Settings(_env_file=None).log_document_content is False


# --- configure_from_settings -------------------------------------------------


def test_configure_from_settings_honours_the_level() -> None:
    stream, log = configure_from_settings_and_capture(Settings(_env_file=None, log_level="ERROR"))

    log.info("not shown")
    log.error("shown")

    output = stream.getvalue()
    assert "shown" in output
    assert "not shown" not in output


def test_configure_from_settings_honours_the_content_flag() -> None:
    """The documented local-debugging escape hatch actually works."""
    stream, _ = configure_from_settings_and_capture(
        Settings(_env_file=None, log_document_content=True)
    )
    log = get_logger("test.probe")

    log.info("file loaded", content=SECRET)

    assert SECRET in stream.getvalue()


def test_configure_from_settings_redacts_by_default() -> None:
    stream, _ = configure_from_settings_and_capture(Settings(_env_file=None))
    log = get_logger("test.probe")

    log.info("file loaded", content=SECRET)

    assert SECRET not in stream.getvalue()


# --- get_logger --------------------------------------------------------------


def test_get_logger_returns_something_that_can_log() -> None:
    log = get_logger("app.test")

    assert hasattr(log, "info")
    assert hasattr(log, "warning")
    assert hasattr(log, "error")
