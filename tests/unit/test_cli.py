"""Tests for the command-line entry points (:mod:`scripts.cli`).

What matters about a CLI is not that it calls the right functions but that
a person — or a shell script — can act on the result, so most of these
tests assert on exit statuses, on what lands on stdout versus stderr, and
on the deliberate distinction between "your invocation is wrong" (status 1)
and "your data is wrong" (status 3).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
import structlog

from app.config import Settings
from scripts import cli
from scripts.samples import prepare


@pytest.fixture(autouse=True)
def restore_logging() -> object:
    """Put structlog and the root logger back after every test.

    ``main()`` configures logging for real, which is the behaviour under
    test — and also global state that would otherwise leak into every other
    test in the session.
    """
    saved_config = structlog.get_config()
    root = logging.getLogger()
    saved_level = root.level
    saved_handlers = list(root.handlers)
    yield
    structlog.configure(**saved_config)  # type: ignore[arg-type]
    root.setLevel(saved_level)
    root.handlers[:] = saved_handlers


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """A prepared sample corpus, which deliberately contains failures."""
    destination = tmp_path / "samples"
    prepare(destination)
    return destination


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------
def test_prepare_writes_the_corpus(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The default command creates the sample corpus and says what it wrote."""
    destination = tmp_path / "samples"

    status = cli.main(["prepare", "--dest", str(destination)])

    printed = capsys.readouterr().out
    assert status == cli.EXIT_OK
    assert "+ created" in printed
    assert "corpus/api-spec.md" in printed
    assert (destination / "corpus" / "api-spec.md").is_file()
    assert (destination / "edge-cases" / "logo.png").is_file()


def test_prepare_json_is_machine_readable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--format json`` puts nothing but JSON on stdout, for a script to read."""
    destination = tmp_path / "samples"

    status = cli.main(["prepare", "--dest", str(destination), "--format", "json"])

    payload = json.loads(capsys.readouterr().out)
    assert status == cli.EXIT_OK
    assert payload["total"] == 10
    assert len(payload["created"]) == 10
    assert payload["kept"] == []


def test_prepare_second_run_reports_unchanged(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Running twice is free: nothing is rewritten, and it is reported."""
    destination = tmp_path / "samples"
    cli.main(["prepare", "--dest", str(destination)])
    capsys.readouterr()

    status = cli.main(["prepare", "--dest", str(destination)])

    printed = capsys.readouterr().out
    assert status == cli.EXIT_OK
    assert "= unchanged" in printed
    assert "+ created" not in printed


def test_prepare_keeps_edited_files_without_force(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An edited sample is reported, not overwritten, and the exit status stays 0.

    Nothing went wrong: the corpus was written, one file was deliberately
    left alone. Status 3 would tell a caller the data is broken, which it is
    not — someone simply chose their own content.
    """
    destination = tmp_path / "samples"
    cli.main(["prepare", "--dest", str(destination)])
    edited = destination / "corpus" / "faq.txt"
    edited.write_text("mine", encoding="utf-8")

    status = cli.main(["prepare", "--dest", str(destination)])

    printed = capsys.readouterr().out
    assert status == cli.EXIT_OK
    assert "! kept" in printed
    assert "--force" in printed
    assert edited.read_text(encoding="utf-8") == "mine"


def test_prepare_force_replaces_edited_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--force`` is the only thing that overwrites."""
    destination = tmp_path / "samples"
    cli.main(["prepare", "--dest", str(destination)])
    edited = destination / "corpus" / "faq.txt"
    edited.write_text("mine", encoding="utf-8")

    status = cli.main(["prepare", "--dest", str(destination), "--force"])

    printed = capsys.readouterr().out
    assert status == cli.EXIT_OK
    assert "~ updated" in printed
    assert "! kept" not in printed
    assert edited.read_text(encoding="utf-8") != "mine"


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------
def test_ingest_of_an_empty_directory_says_so(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A directory with nothing in it is a success, not an error."""
    empty = tmp_path / "empty"
    empty.mkdir()

    status = cli.main(["ingest", str(empty)])

    printed = capsys.readouterr().out
    assert status == cli.EXIT_OK
    assert "no files found" in printed


def test_ingest_over_the_corpus_exits_three_because_files_failed(
    corpus: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The report still prints, with the codes that explain the status.

    Exiting non-zero is what lets a pipeline notice broken input, but a
    status with no report behind it would leave the caller guessing which
    files to fix.
    """
    status = cli.main(["ingest", str(corpus)])

    printed = capsys.readouterr().out
    assert status == cli.EXIT_FILES_FAILED
    assert "pdf_encrypted" in printed
    assert "binary_content" in printed
    assert "loaded" in printed


def test_ingest_json_is_pure_json_on_stdout(
    corpus: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """stdout carries the report; anything else would break a pipe."""
    status = cli.main(["ingest", str(corpus), "--format", "json"])

    payload = json.loads(capsys.readouterr().out)
    assert status == cli.EXIT_FILES_FAILED
    assert payload["total_files"] == 10
    assert payload["loaded"] == 5
    assert payload["unsupported"] == 1
    assert {f["code"] for f in payload["failures"]} == {"binary_content", "pdf_encrypted"}


def test_ingest_prints_the_directory_name_not_the_absolute_path(
    corpus: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Reports name the batch by its basename, as every other report does.

    The same report will later be printed by the API and written to logs,
    so the CLI is not allowed to be the one place a full path escapes.
    """
    cli.main(["ingest", str(corpus)])

    printed = capsys.readouterr().out
    assert str(corpus) not in printed
    assert corpus.name in printed


def test_ingest_of_a_missing_directory_explains_itself_without_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A deliberate error prints a message and a code, and exits 1."""
    status = cli.main(["ingest", str(tmp_path / "not-there")])

    errors = capsys.readouterr().err
    assert status == cli.EXIT_DID_NOT_RUN
    assert "error:" in errors
    assert "discovery_root_not_found" in errors
    assert "Traceback" not in errors


def test_ingest_default_path_is_the_configured_samples_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Called with no path, the command reads the sample corpus."""
    destination = tmp_path / "my-samples"
    prepare(destination)
    monkeypatch.setattr(
        cli,
        "get_settings",
        lambda: Settings(_env_file=None, sample_docs_dir=destination),
    )

    status = cli.main(["ingest"])

    assert status == cli.EXIT_FILES_FAILED  # the samples include failures
    assert "my-samples" in capsys.readouterr().out


def test_ingest_without_samples_tells_you_how_to_make_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A fresh clone has no corpus yet, so the hint must name the command."""
    monkeypatch.setattr(
        cli,
        "get_settings",
        lambda: Settings(_env_file=None, sample_docs_dir=tmp_path / "not-there"),
    )

    status = cli.main(["ingest"])

    errors = capsys.readouterr().err
    assert status == cli.EXIT_DID_NOT_RUN
    assert "scripts.cli prepare" in errors
    assert "Traceback" not in errors


def test_no_recursive_reads_only_the_top_level(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The flag is passed through to discovery rather than being decorative."""
    root = tmp_path / "docs"
    nested = root / "nested"
    nested.mkdir(parents=True)
    (root / "one.txt").write_text("alpha", encoding="utf-8")
    (nested / "two.txt").write_text("beta", encoding="utf-8")

    cli.main(["ingest", str(root), "--no-recursive", "--format", "json"])
    flat = json.loads(capsys.readouterr().out)

    cli.main(["ingest", str(root), "--format", "json"])
    recursive = json.loads(capsys.readouterr().out)

    assert flat["total_files"] == 1
    assert recursive["total_files"] == 2


# ---------------------------------------------------------------------------
# The command surface itself
# ---------------------------------------------------------------------------
def test_parser_accepts_both_commands_and_their_flags() -> None:
    """Both commands exist, and the shared flags reach each of them."""
    parser = cli.build_parser()

    assert parser.parse_args(["prepare"]).command == "prepare"
    assert parser.parse_args(["ingest"]).command == "ingest"
    assert parser.parse_args(["prepare"]).format == "text"
    assert parser.parse_args(["prepare", "--format", "json"]).format == "json"
    assert parser.parse_args(["ingest", "--no-recursive"]).no_recursive is True


def test_usage_errors_are_argparses_own_exit_status() -> None:
    """Bad arguments stop before any work, with status 2."""
    with pytest.raises(SystemExit) as no_command:
        cli.main([])

    with pytest.raises(SystemExit) as unknown:
        cli.main(["frobnicate"])

    assert no_command.value.code == 2
    assert unknown.value.code == 2


def test_help_exits_zero_and_prints_usage(capsys: pytest.CaptureFixture[str]) -> None:
    """``--help`` is a successful run, not an error."""
    with pytest.raises(SystemExit) as help_run:
        cli.main(["--help"])

    assert help_run.value.code == 0
    assert "usage" in capsys.readouterr().out


def test_exit_statuses_are_the_documented_values() -> None:
    """Pinned so a refactor cannot quietly renumber them under a caller."""
    assert cli.EXIT_OK == 0
    assert cli.EXIT_DID_NOT_RUN == 1
    assert cli.EXIT_FILES_FAILED == 3


# ---------------------------------------------------------------------------
# How the CLI behaves as a citizen of the rest of the system
# ---------------------------------------------------------------------------
def test_an_error_without_detail_prints_no_detail_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Detail is optional on :class:`RAGError`; the CLI degrades to message + code.

    A bare error is what a subclass raises when it has nothing to add, so
    the CLI has to be correct here and not only when discovery fills every
    field in. The default code still reaches the caller, which is the point
    of printing it at all.
    """

    def raise_bare(*_args: object, **_kwargs: object) -> None:
        from app.domain.errors import RAGError

        raise RAGError("something went wrong")

    monkeypatch.setattr(cli, "ingest", raise_bare)

    status = cli.main(["ingest", str(tmp_path)])

    errors = capsys.readouterr().err
    assert status == cli.EXIT_DID_NOT_RUN
    assert errors.splitlines() == [
        "error: something went wrong",
        "  code: internal_error",
    ]


def test_unexpected_errors_are_not_swallowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bug keeps its traceback; only deliberate errors are caught."""
    monkeypatch.setattr(cli, "ingest", lambda *_args, **_kwargs: 1 / 0)

    with pytest.raises(ZeroDivisionError):
        cli.main(["ingest", str(tmp_path)])


def test_logging_is_configured_at_startup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Everything logged on the way is structured and redacted.

    The CLI must not be the one place document content could reach a
    terminal, so configuration happens before any pipeline work.
    """
    seen: list[Settings] = []
    monkeypatch.setattr(cli, "configure_from_settings", seen.append)

    status = cli.main(["prepare", "--dest", str(tmp_path / "samples")])

    assert status == cli.EXIT_OK
    assert [type(s) for s in seen] == [Settings]
