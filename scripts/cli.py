"""Command-line entry points for the ingestion pipeline.

Two commands, because they are the two things a person needs before the
later stages exist:

``prepare``
    Write the sample corpus into the configured samples directory. This is
    what turns a fresh clone into something the pipeline can read, which is
    why ``ingest`` defaults to that same directory.

``ingest``
    Run discovery, loading and metadata over a directory and print the
    report the reporting stage already knows how to build. The wall time
    measured here is the report's real duration, not an estimate.

Usage::

    python -m scripts.cli prepare [--dest PATH] [--force] [--format text|json]
    python -m scripts.cli ingest [PATH] [--no-recursive] [--format text|json]

Exit statuses
------------
0
    The command ran and no file failed.
1
    The command could not run: a missing directory, an unreadable one.
2
    argparse rejected the arguments (its own convention, not ours).
3
    The command ran and produced a report, but some files failed to load.

The distinction between 1 and 3 is the same one the pipeline keeps
everywhere else — "your invocation is wrong" and "your data is wrong" are
different problems, and a caller must be able to tell them apart. The
report still prints on status 3; there is nothing to act on otherwise.

Deliberate errors (:class:`RAGError`) print a message and a code instead of
a traceback. Anything else keeps its traceback, because a bug must look
like a bug.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from app.config import Settings, get_settings
from app.domain.errors import RAGError
from app.ingestion.reporting import build_report
from app.ingestion.service import ingest
from app.observability.logging import configure_from_settings, get_logger
from scripts.samples import prepare

__all__ = [
    "EXIT_DID_NOT_RUN",
    "EXIT_FILES_FAILED",
    "EXIT_OK",
    "build_parser",
    "main",
]

EXIT_OK = 0
EXIT_DID_NOT_RUN = 1
EXIT_FILES_FAILED = 3

#: Offered by both commands, so built once and shared via ``parents``.
_FORMATS = ("text", "json")


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returned rather than hidden inside :func:`main` so a test — or a future
    command — can inspect or extend it without parsing anything.
    """
    output = argparse.ArgumentParser(add_help=False)
    output.add_argument(
        "--format",
        choices=_FORMATS,
        default="text",
        help="print the report as readable text (default) or as JSON on stdout",
    )

    parser = argparse.ArgumentParser(
        prog="python -m scripts.cli",
        description="Ingest technical documents and report on what happened.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    prepare_parser = commands.add_parser(
        "prepare",
        parents=[output],
        help="write the sample corpus into the samples directory",
    )
    prepare_parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        metavar="PATH",
        help="where to write (default: the configured samples directory)",
    )
    prepare_parser.add_argument(
        "--force",
        action="store_true",
        help="replace existing files that differ instead of leaving them alone",
    )

    ingest_parser = commands.add_parser(
        "ingest",
        parents=[output],
        help="discover, load and describe every supported file under a directory",
    )
    ingest_parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=None,
        metavar="PATH",
        help="directory to read (default: the configured samples directory)",
    )
    ingest_parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="read only the files directly inside PATH, not its sub-directories",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command and return the process exit status.

    Parameters
    ----------
    argv:
        Arguments *excluding* the program name, exactly like
        :func:`sys.argv`. ``None`` means "use the real command line", so
        ``python -m scripts.cli ...`` and a test calling ``main([...])``
        take the same path.

    Returns
    -------
    int
        One of :data:`EXIT_OK`, :data:`EXIT_DID_NOT_RUN` or
        :data:`EXIT_FILES_FAILED`. A usage error is not returned: argparse
        raises ``SystemExit(2)`` itself.
    """
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    settings = get_settings()
    # Logging is configured so that anything the pipeline logs on the way is
    # structured and redacted — the CLI must not be the one place document
    # content could reach a terminal. The happy path stays quiet, though:
    # the report belongs on stdout and noise on stderr confuses callers, so
    # detail is available with LOG_LEVEL=DEBUG rather than printed always.
    configure_from_settings(settings)

    try:
        if args.command == "prepare":
            return _run_prepare(args, settings)
        return _run_ingest(args, settings)
    except RAGError as exc:
        # Deliberate failure: a message and a code, never a traceback. The
        # code is always present -- RAGError falls back to a default -- so
        # it is printed unconditionally rather than guarded into silence.
        print(f"error: {exc}", file=sys.stderr)
        print(f"  code: {exc.code}", file=sys.stderr)
        if exc.detail:
            print(f"  detail: {exc.detail}", file=sys.stderr)
        return EXIT_DID_NOT_RUN


def _run_prepare(args: argparse.Namespace, settings: Settings) -> int:
    """Write the sample corpus and report what was written."""
    destination = settings.sample_docs_dir if args.dest is None else args.dest

    result = prepare(destination, force=args.force)

    if args.format == "json":
        print(json.dumps(result.summary(), indent=2))
    else:
        print(result.render())
    return EXIT_OK


def _run_ingest(args: argparse.Namespace, settings: Settings) -> int:
    """Run the pipeline over a directory and print its report."""
    root = settings.sample_docs_dir if args.path is None else args.path

    if args.path is None and not root.exists():
        # The default is the sample corpus, which a fresh clone does not
        # have yet. Saying what to run beats a bare "does not exist".
        print(
            f"error: {root.name} does not exist yet\n"
            "  hint: run `python -m scripts.cli prepare` to write the sample corpus",
            file=sys.stderr,
        )
        return EXIT_DID_NOT_RUN

    started = time.perf_counter()
    result = ingest(root, recursive=not args.no_recursive)
    elapsed = time.perf_counter() - started

    report = build_report(result, duration_seconds=elapsed)

    if args.format == "json":
        print(report.to_json())
    else:
        print(report.render())

    # Deliberately DEBUG: a caller piping stdout into another tool must not
    # find this waiting on stderr, but a developer who asks for it should
    # get the real path rather than a basename they have to guess from.
    get_logger(__name__).debug(
        "ingest.finished",
        root=str(root),
        total=result.total_count,
        loaded=result.loaded_count,
        failed=result.failed_count,
        duration_seconds=elapsed,
    )

    return EXIT_FILES_FAILED if report.has_failures else EXIT_OK


if __name__ == "__main__":  # pragma: no cover - the interpreter's entry point
    raise SystemExit(main())
