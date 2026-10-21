"""Structured logging that cannot leak document content by default.

This module has one job beyond "make the logs parseable": an operator reading
a shared log store should never find a document's text in it. Extraction
produces exactly the material that must *not* be copied into logs — a PDF's
full text, a chunk, an assembled answer — and it is one careless
``logger.info("loaded", text=doc.content)`` away from being retained forever
in a system with a much longer memory than the index it was debugging.

So redaction is **on by default** and has to be turned *off* deliberately,
never on.

Two layers, because neither is sufficient alone
-----------------------------------------------

1. **A key-name deny-list.** Anything logged under a name like ``content``,
   ``text`` or ``page_content`` is replaced with ``<redacted>`` — even a
   short value, because a one-line secret is still a secret. This catches
   the mistakes people actually make.
2. **A value-size guard.** Any string longer than ``max_value_chars`` is
   replaced with ``<omitted N chars>`` *whatever it is called under*. This
   is the backstop for the mistakes the deny-list cannot see: a future
   developer invents a new key name, and a 1000-character chunk still cannot
   arrive intact. The length is kept because "why is this field 4,000
   characters?" is usually the useful part of the answer.

An omitted value reports its size rather than a truncation preview. A
preview would be content too, and a half-leaked document is still a leaked
document.

Arbitrary objects are collapsed to their type name
--------------------------------------------------

A third hole matters as much as the first two: :class:`Document` is a frozen
dataclass whose ``repr`` includes its own ``content``. Handing one to a
logger would smuggle the whole text past both layers above, because the
value is neither a long string nor a flagged key — it is an object whose
``repr`` happens to contain the text. So unrecognised objects are rendered
as ``<Document>`` instead. The escape hatch for wanting their contents is
the same as everywhere else in this project: call ``.summary()``, which is
built to be safe to emit.

``str``, numbers, booleans, ``None``, :class:`pathlib.Path`,
:class:`enum.Enum` and dates survive untouched — logging a path or an
outcome is ordinary and harmless.

The escape hatch
----------------

``allow_document_content=True`` disables all three layers. It exists for
local debugging and is off unless ``LOG_DOCUMENT_CONTENT`` says otherwise
(``Settings`` defaults it to ``False``), so the safe behaviour is the one a
fresh install gets. There is no warning banner when it is on; the config
value is the record.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal, cast

import structlog
from structlog.types import EventDict, Processor

from app.config import Settings

__all__ = [
    "CONTENT_KEYS",
    "DEFAULT_MAX_VALUE_CHARS",
    "REDACTED",
    "configure_from_settings",
    "configure_logging",
    "get_logger",
    "make_content_redactor",
]

#: Key names whose value is document text. Deliberately explicit rather than
#: a substring match: ``text_file_count=5`` must survive redaction intact, and
#: a broad ``"text" in key`` rule would happily swallow that count.
CONTENT_KEYS = frozenset(
    {
        "answer",
        "answer_text",
        "body",
        "chunk",
        "chunk_text",
        "chunks",
        "content",
        "content_preview",
        "document",
        "document_text",
        "excerpt",
        "page_content",
        "page_contents",
        "query",
        "raw",
        "raw_text",
        "text",
    }
)

#: Replacement for a value withheld by the key-name deny-list.
REDACTED = "<redacted>"

#: Strings longer than this are withheld by the size guard. Below the default
#: chunk size (1000) on purpose: a whole chunk must never fit.
DEFAULT_MAX_VALUE_CHARS = 512

#: How deep nested dicts/lists are walked before being collapsed. Shallow
#: enough to bound the work, deep enough for realistic payloads.
MAX_NESTING_DEPTH = 4

_LEVELS: dict[str, int] = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "NOTSET": logging.NOTSET,
}


def _is_content_key(key: object) -> bool:
    """True when a value logged under this key is document text."""
    return isinstance(key, str) and key.lower() in CONTENT_KEYS


def _scrub(value: Any, *, max_chars: int, depth: int) -> Any:
    """Apply both redaction layers to one value, descending into containers.

    Returns a value that is always safe to serialise: strings are bounded,
    flagged keys are blanked, and anything unrecognised is reduced to its
    type name rather than being ``repr``-ed.
    """
    if depth > MAX_NESTING_DEPTH:
        return "<omitted: nested too deeply>"

    if isinstance(value, Path):
        # As a plain string, so the renderer need not repr() it.
        return str(value)

    if isinstance(value, Enum):
        # Re-dispatch on the underlying value: it gets the same size guard as
        # any other string, and json.dumps never sees an object it would have
        # to fall back to repr() for.
        return _scrub(value.value, max_chars=max_chars, depth=depth + 1)

    if isinstance(value, (str, int, float, bool, type(None))):
        if isinstance(value, str) and len(value) > max_chars:
            return f"<omitted {len(value)} chars>"
        return value

    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"<omitted {len(value)} bytes>"

    if isinstance(value, Mapping):
        return {
            str(key): (
                REDACTED
                if _is_content_key(key)
                else _scrub(item, max_chars=max_chars, depth=depth + 1)
            )
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_scrub(item, max_chars=max_chars, depth=depth + 1) for item in value]

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    # Anything else: an object whose repr may quote document text (a Document
    # does exactly this). Name the type instead of the contents.
    return f"<{type(value).__name__}>"


def make_content_redactor(
    *,
    allow_document_content: bool = False,
    max_value_chars: int = DEFAULT_MAX_VALUE_CHARS,
) -> Processor:
    """Build the redaction processor, for wiring into a structlog pipeline.

    Parameters
    ----------
    allow_document_content:
        When true the processor is a pass-through, and document text may be
        logged. Off by default; see the module docstring.
    max_value_chars:
        Size guard threshold. Shorter values survive; longer ones are
        replaced by a note carrying their length.

    Returns
    -------
    structlog.types.Processor
        Ready to place anywhere in a processor chain — but *before* the
        renderer, so the renderer never sees the withheld values.
    """
    if max_value_chars < 0:
        raise ValueError(f"max_value_chars must be >= 0, got {max_value_chars}")

    def redact(_logger: object, _method_name: str, event_dict: EventDict) -> EventDict:
        if allow_document_content:
            return event_dict
        # _scrub is Any-typed because it descends into arbitrary values, but
        # fed a Mapping it always returns a dict — which is all EventDict is.
        return cast(EventDict, _scrub(event_dict, max_chars=max_value_chars, depth=0))

    return redact


def configure_logging(
    *,
    level: str = "INFO",
    allow_document_content: bool = False,
    renderer: Literal["json", "console"] = "json",
    max_value_chars: int = DEFAULT_MAX_VALUE_CHARS,
) -> None:
    """Configure structlog for the whole process.

    Safe to call more than once — it replaces the previous configuration
    rather than layering a second one on top, which is what makes the
    reconfiguration fixtures in the tests possible.

    ``level`` is applied to bound loggers *and* to the stdlib root logger, so
    records arriving from third-party libraries (pypdf, uvicorn) obey the
    same threshold rather than quietly out-ranking it.

    Parameters
    ----------
    level:
        One of ``CRITICAL``, ``ERROR``, ``WARNING``, ``INFO``, ``DEBUG``,
        ``NOTSET``. Case-insensitive, like ``Settings.LOG_LEVEL``.
    allow_document_content:
        Disables redaction. See the module docstring.
    renderer:
        ``json`` for machine-readable output, ``console`` for a readable
        local session. The redaction layers are identical either way.
    max_value_chars:
        Size guard threshold, forwarded to the redactor.
    """
    level_value = _coerce_level(level)

    processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        # Before the renderer, so withheld values never reach it.
        make_content_redactor(
            allow_document_content=allow_document_content,
            max_value_chars=max_value_chars,
        ),
    ]
    if renderer == "json":
        processors.append(structlog.processors.JSONRenderer())
    elif renderer == "console":
        processors.append(structlog.dev.ConsoleRenderer())
    else:  # pragma: no cover - guarded by the Literal annotation
        raise ValueError(f"unknown renderer {renderer!r}; use 'json' or 'console'")

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(level_value),
        logger_factory=structlog.stdlib.LoggerFactory(),
        # False so a test can reconfigure and see the new processors: with
        # caching on, a logger built under the old config keeps it forever.
        cache_logger_on_first_use=False,
    )

    root = logging.getLogger()
    root.setLevel(level_value)
    if not root.handlers:
        # Under pytest the root already has capture handlers and adding one
        # would fight with them; in a real process there is nothing to
        # inherit, so records would otherwise be silently dropped.
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        root.addHandler(handler)


def configure_from_settings(settings: Settings) -> None:
    """Configure logging from the typed settings object.

    Uses the JSON renderer: this is the startup path, and a process whose
    destination is not yet known should emit the structured form. Local
    console output is available by calling
    :func:`configure_logging` directly.
    """
    configure_logging(
        level=settings.log_level,
        allow_document_content=settings.log_document_content,
    )


def get_logger(name: str | None = None) -> Any:
    """Return a logger. Import this rather than calling ``structlog``.

    A single import site, matching how :mod:`app.ingestion.loaders`
    re-exports ``load_pdf``: callers should not need to know which logging
    library is underneath when it is swapped.
    """
    return structlog.get_logger(name)


def _coerce_level(level: str) -> int:
    """Turn a level name into a logging level, rejecting nonsense.

    ``Settings`` validates ``LOG_LEVEL`` at startup, but this is also called
    directly, so the same six names are enforced here rather than trusting
    the caller.
    """
    normalised = level.strip().upper()
    if normalised not in _LEVELS:
        raise ValueError(f"unknown log level {level!r}; expected one of {sorted(_LEVELS)}")
    return _LEVELS[normalised]
