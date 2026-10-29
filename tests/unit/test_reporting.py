"""Tests for the per-batch ingest report and the index report.

A report has two audiences — a person reading it and a machine parsing it —
and one hard constraint: it may never carry document text. Most tests below
therefore assert on *structure* (counts, groupings, determinism), with a
recurring explicit check that a document's content does not appear in any
rendered form, because that is the property the summary methods it derives
from already guarantee and this module must not be the one to break it.

Batches are built synthetically for the grouping and ordering tests: a report
should be reproducible from a handful of hand-made results, without a
filesystem. Two end-to-end tests at the end run against a real ``ingest()``
so the derivation is proven against actual output rather than only against
expectations; the index report gets the same treatment against a real
``index_directory()`` pass.
"""

from __future__ import annotations

import dataclasses
import json
import tempfile
from pathlib import Path

import pytest

from app.domain.models import IngestStatus
from app.ingestion.deduplication import ChangeKind, DedupDecision, classify
from app.ingestion.discovery import UnsupportedFile
from app.ingestion.indexer import IndexOutcome, IndexResult, index_directory
from app.ingestion.pipeline import PipelineAction
from app.ingestion.reporting import (
    UNSPECIFIED_CODE,
    CodeTally,
    build_index_report,
    build_report,
    format_bytes,
)
from app.ingestion.service import FileOutcome, FileResult, IngestResult, ingest
from app.storage.registry import DocumentRegistry
from tests.fakes import FakeChunkStore, FakeEmbedder

SECRET = "confidential clause nine must not reach a report"


def file_result(
    path: str,
    outcome: FileOutcome,
    *,
    code: str | None = None,
    size: int | None = None,
    reason: str | None = None,
) -> FileResult:
    """Build one synthetic result with only the fields a report reads."""
    return FileResult(
        source_name=path.rsplit("/", 1)[-1],
        relative_path=path,
        media_type="text/markdown",
        byte_size=size,
        outcome=outcome,
        code=code,
        reason=reason,
    )


def batch(
    files: list[FileResult],
    *,
    root: str = "docs",
    unsupported: list[UnsupportedFile] | None = None,
) -> IngestResult:
    """Assemble synthetic results into a batch."""
    return IngestResult(
        root=root,
        files=tuple(files),
        unsupported=tuple(unsupported or ()),
    )


def loaded(path: str, *, size: int | None = 100) -> FileResult:
    return file_result(path, FileOutcome.LOADED, size=size)


# --- counts ------------------------------------------------------------------


def test_counts_mirror_the_batch() -> None:
    result = batch(
        [
            loaded("a.md"),
            loaded("b.md"),
            file_result("c.md", FileOutcome.SKIPPED, code="empty_file", size=0),
            file_result("d.md", FileOutcome.FAILED, code="binary_content", size=None),
        ],
        unsupported=[UnsupportedFile("logo.png", "logo.png", ".png", "unsupported")],
    )

    report = build_report(result)

    assert report.total_files == 5
    assert report.processed_files == 4
    assert report.loaded == 2
    assert report.skipped == 1
    assert report.failed == 1
    assert report.unsupported == 1


def test_indexable_equals_the_loaded_count() -> None:
    """Files with text ready to index are exactly the ones that loaded."""
    result = batch([loaded("a.md"), loaded("b.md")])

    assert build_report(result).indexable == 2


def test_total_problems_is_failed_plus_skipped() -> None:
    result = batch(
        [
            loaded("a.md"),
            file_result("b.md", FileOutcome.SKIPPED, code="empty_file"),
            file_result("c.md", FileOutcome.FAILED, code="binary_content"),
        ]
    )

    report = build_report(result)

    assert report.total_problems == 2
    assert report.has_failures


def test_is_empty_for_a_batch_with_no_files() -> None:
    report = build_report(batch([]))

    assert report.is_empty
    assert not report.has_failures


def test_has_failures_tracks_the_failed_count() -> None:
    ok = build_report(batch([loaded("a.md")]))
    bad = build_report(batch([file_result("a.md", FileOutcome.FAILED, code="file_not_found")]))

    assert not ok.has_failures
    assert bad.has_failures


# --- bytes read --------------------------------------------------------------


def test_total_bytes_sums_the_sizes_learned() -> None:
    result = batch([loaded("a.md", size=100), loaded("b.md", size=50)])

    assert build_report(result).total_bytes == 150


def test_total_bytes_is_none_when_nothing_reported_a_size() -> None:
    """Not ``0``: "never measured" and "there was nothing" differ."""
    result = batch([file_result("a.md", FileOutcome.FAILED, size=None)])

    assert build_report(result).total_bytes is None


def test_total_bytes_is_none_for_an_empty_batch() -> None:
    assert build_report(batch([])).total_bytes is None


def test_total_bytes_adds_the_sizes_it_has_and_ignores_the_rest() -> None:
    result = batch(
        [
            loaded("a.md", size=10),
            file_result("b.md", FileOutcome.FAILED, size=None),
            loaded("c.md", size=5),
        ]
    )

    assert build_report(result).total_bytes == 15


# --- problem grouping --------------------------------------------------------


def test_failures_are_bucketed_by_code() -> None:
    result = batch(
        [
            file_result("a.pdf", FileOutcome.FAILED, code="encrypted_pdf"),
            file_result("b.pdf", FileOutcome.FAILED, code="encrypted_pdf"),
            file_result("c.md", FileOutcome.FAILED, code="binary_content"),
        ]
    )

    report = build_report(result)

    assert [(t.code, t.count) for t in report.failures] == [
        ("encrypted_pdf", 2),
        ("binary_content", 1),
    ]


def test_each_tally_carries_the_paths_that_produced_it() -> None:
    result = batch(
        [
            file_result("a.pdf", FileOutcome.FAILED, code="encrypted_pdf"),
            file_result("b.pdf", FileOutcome.FAILED, code="encrypted_pdf"),
        ]
    )

    tally = build_report(result).failures[0]

    assert tally.paths == ("a.pdf", "b.pdf")
    assert tally.count == len(tally.paths)


def test_paths_within_a_code_stay_in_discovery_order() -> None:
    """Order a reader can match against the directory listing."""
    result = batch(
        [
            file_result("z.md", FileOutcome.SKIPPED, code="empty_file"),
            file_result("a.md", FileOutcome.SKIPPED, code="empty_file"),
            file_result("m.md", FileOutcome.SKIPPED, code="empty_file"),
        ]
    )

    assert build_report(result).skips[0].paths == ("z.md", "a.md", "m.md")


def test_tallies_are_ordered_by_frequency_then_code() -> None:
    """Deterministic: ties break alphabetically, so two runs match."""
    result = batch(
        [
            file_result("a.md", FileOutcome.FAILED, code="zebra_error"),
            file_result("b.md", FileOutcome.FAILED, code="alpha_error"),
            file_result("c.md", FileOutcome.FAILED, code="zebra_error"),
            file_result("d.md", FileOutcome.FAILED, code="alpha_error"),
        ]
    )

    assert [t.code for t in build_report(result).failures] == [
        "alpha_error",
        "zebra_error",
    ]


def test_skips_are_grouped_separately_from_failures() -> None:
    result = batch(
        [
            file_result("a.md", FileOutcome.SKIPPED, code="empty_file"),
            file_result("b.pdf", FileOutcome.FAILED, code="encrypted_pdf"),
        ]
    )

    report = build_report(result)

    assert [t.code for t in report.skips] == ["empty_file"]
    assert [t.code for t in report.failures] == ["encrypted_pdf"]


def test_a_result_without_a_code_is_labelled_unspecified() -> None:
    """The type allows ``None``; the report must not crash on it."""
    result = batch([file_result("a.md", FileOutcome.SKIPPED, code=None)])

    report = build_report(result)

    assert [t.code for t in report.skips] == [UNSPECIFIED_CODE]


def test_no_failures_means_an_empty_tuple_not_a_placeholder() -> None:
    report = build_report(batch([loaded("a.md")]))

    assert report.failures == ()
    assert report.skips == ()


# --- deduplication tallies ---------------------------------------------------


def decision(kind: ChangeKind, path: str = "a.md") -> DedupDecision:
    return DedupDecision(kind=kind, doc_id="doc_" + "0" * 16, content_sha256="0" * 64)


def test_dedup_is_none_when_the_caller_supplied_no_decisions() -> None:
    """Deduplication not having run must not render as all-zeros."""
    report = build_report(batch([loaded("a.md")]))

    assert report.dedup is None


def test_dedup_counts_decisions_per_kind() -> None:
    report = build_report(
        batch([loaded("a.md")]),
        decisions=[decision(ChangeKind.NEW), decision(ChangeKind.NEW)],
    )

    assert dict(report.dedup)["new"] == 2  # type: ignore[arg-type]


def test_dedup_keeps_zero_counts_for_absent_kinds() -> None:
    """A report omitting ``renamed`` and one saying ``renamed 0`` differ."""
    report = build_report(
        batch([loaded("a.md")]),
        decisions=[decision(ChangeKind.NEW)],
    )

    assert dict(report.dedup) == {  # type: ignore[arg-type]
        "new": 1,
        "unchanged": 0,
        "modified": 0,
        "renamed": 0,
    }


def test_dedup_follows_change_kind_declaration_order() -> None:
    kinds = [kind.value for kind in ChangeKind]
    report = build_report(
        batch([loaded("a.md")]),
        decisions=[decision(ChangeKind.RENAMED)],
    )

    assert [pair[0] for pair in report.dedup] == kinds  # type: ignore[index]


def test_an_empty_decision_list_is_not_the_same_as_no_deduplication() -> None:
    report = build_report(batch([loaded("a.md")]), decisions=[])

    assert report.dedup is not None
    assert all(count == 0 for _, count in report.dedup)


def test_dedup_decisions_need_not_match_the_file_count() -> None:
    """Counts only — a subset (attempted files, say) is legitimate."""
    report = build_report(
        batch([loaded("a.md"), loaded("b.md"), loaded("c.md")]),
        decisions=[decision(ChangeKind.NEW)],
    )

    assert dict(report.dedup)["new"] == 1  # type: ignore[arg-type]


# --- duration ----------------------------------------------------------------


def test_duration_is_carried_through() -> None:
    report = build_report(batch([loaded("a.md")]), duration_seconds=1.5)

    assert report.duration_seconds == 1.5


def test_duration_defaults_to_none_because_a_report_must_not_invent_one() -> None:
    assert build_report(batch([loaded("a.md")])).duration_seconds is None


def test_a_negative_duration_is_rejected() -> None:
    """Only a broken clock produces it; printing would launder the error."""
    with pytest.raises(ValueError, match="duration_seconds"):
        build_report(batch([loaded("a.md")]), duration_seconds=-1.0)


# --- machine-readable output -------------------------------------------------


def test_to_dict_is_json_serialisable() -> None:
    json.dumps(build_report(batch([loaded("a.md")])).to_dict())


def test_to_dict_carries_the_counts_and_groupings() -> None:
    result = batch(
        [file_result("a.md", FileOutcome.FAILED, code="binary_content")],
        unsupported=[UnsupportedFile("logo.png", "logo.png", ".png", "unsupported")],
    )

    payload = build_report(result).to_dict()

    assert payload["failed"] == 1
    assert payload["unsupported"] == 1
    assert payload["failures"] == [{"code": "binary_content", "count": 1, "paths": ["a.md"]}]


def test_dedup_key_is_omitted_entirely_when_not_supplied() -> None:
    payload = build_report(batch([loaded("a.md")])).to_dict()

    assert "dedup" not in payload


def test_dedup_key_is_a_mapping_when_supplied() -> None:
    payload = build_report(
        batch([loaded("a.md")]),
        decisions=[decision(ChangeKind.MODIFIED)],
    ).to_dict()

    assert payload["dedup"] == {
        "new": 0,
        "unchanged": 0,
        "modified": 1,
        "renamed": 0,
    }


def test_to_json_round_trips_back_to_the_payload() -> None:
    report = build_report(batch([loaded("a.md")]))

    assert json.loads(report.to_json()) == report.to_dict()


def test_to_json_accepts_an_indent_argument() -> None:
    report = build_report(batch([loaded("a.md")]))

    assert "\n" in report.to_json(indent=2)
    assert "\n" not in report.to_json(indent=None)  # type: ignore[arg-type]


def test_the_payload_contains_no_document_text() -> None:
    """The property every summary in this project shares."""
    result = batch(
        [
            file_result(
                "a.md",
                FileOutcome.LOADED,
                code=None,
                size=10,
                reason=SECRET,
            )
        ]
    )
    report = build_report(result)

    assert SECRET not in json.dumps(report.to_dict())
    assert SECRET not in report.to_json()
    assert SECRET not in report.render()


# --- human-readable rendering ------------------------------------------------


def test_render_starts_with_the_report_title_and_root() -> None:
    assert build_report(batch([], root="docs")).render().startswith("Ingest report: docs")


def test_render_summarises_discovered_attempted_and_indexable() -> None:
    report = build_report(batch([loaded("a.md"), loaded("b.md")]))

    assert "2 discovered | 2 attempted | 2 ready to index" in report.render()


def test_render_lists_each_outcome_count() -> None:
    result = batch(
        [
            loaded("a.md"),
            file_result("b.md", FileOutcome.SKIPPED, code="empty_file"),
            file_result("c.md", FileOutcome.FAILED, code="binary_content"),
        ]
    )

    rendered = build_report(result).render()

    assert "loaded       1" in rendered
    assert "skipped      1" in rendered
    assert "failed       1" in rendered


def test_render_says_no_files_found_for_an_empty_batch() -> None:
    rendered = build_report(batch([])).render()

    assert "no files found" in rendered
    assert "ready to index" not in rendered


def test_render_has_no_trailing_newline() -> None:
    """``print()`` supplies the one a terminal wants."""
    assert not build_report(batch([loaded("a.md")])).render().endswith("\n")


def test_render_groups_problems_under_their_codes_with_paths() -> None:
    result = batch(
        [
            file_result("a.pdf", FileOutcome.FAILED, code="encrypted_pdf"),
            file_result("b.pdf", FileOutcome.FAILED, code="encrypted_pdf"),
        ]
    )

    rendered = build_report(result).render()

    assert "failed by code" in rendered
    assert "encrypted_pdf  x2" in rendered
    assert "      a.pdf" in rendered
    assert "      b.pdf" in rendered


def test_render_omits_problem_sections_when_there_are_none() -> None:
    rendered = build_report(batch([loaded("a.md")])).render()

    assert "failed by code" not in rendered
    assert "skipped by code" not in rendered


def test_render_includes_bytes_read() -> None:
    rendered = build_report(batch([loaded("a.md", size=2048)])).render()

    assert "2.0 KiB read" in rendered


def test_render_says_size_unknown_when_nothing_was_measured() -> None:
    rendered = build_report(batch([file_result("a.md", FileOutcome.FAILED, size=None)])).render()

    assert "size unknown" in rendered


def test_render_omits_the_duration_when_not_supplied() -> None:
    """The detail line is the size alone, with no timing suffix to read."""
    rendered = build_report(batch([loaded("a.md", size=100)])).render()

    assert rendered.splitlines()[2] == "  100 B read"


def test_render_formats_the_duration_to_two_decimals() -> None:
    rendered = build_report(batch([loaded("a.md")]), duration_seconds=1.23456).render()

    assert "1.23s" in rendered


def test_render_includes_a_deduplication_section_only_when_supplied() -> None:
    without = build_report(batch([loaded("a.md")])).render()
    with_decisions = build_report(
        batch([loaded("a.md")]), decisions=[decision(ChangeKind.NEW)]
    ).render()

    assert "deduplication" not in without
    assert "deduplication" in with_decisions
    assert "new 1, unchanged 0, modified 0, renamed 0" in with_decisions


def test_render_is_deterministic_for_the_same_batch() -> None:
    result = batch(
        [
            file_result("b.md", FileOutcome.FAILED, code="beta"),
            file_result("a.md", FileOutcome.FAILED, code="alpha"),
        ]
    )

    assert build_report(result).render() == build_report(result).render()


def test_the_report_is_immutable() -> None:
    """Frozen, like every other model — a report cannot shift mid-render."""
    report = build_report(batch([loaded("a.md")]))

    with pytest.raises(dataclasses.FrozenInstanceError):
        report.loaded = 99  # type: ignore[misc]


def test_a_code_tally_is_immutable_and_serialisable() -> None:
    tally = CodeTally(code="empty_file", count=1, paths=("a.md",))

    assert tally.to_dict() == {
        "code": "empty_file",
        "count": 1,
        "paths": ["a.md"],
    }
    with pytest.raises(dataclasses.FrozenInstanceError):
        tally.count = 2  # type: ignore[misc]


# --- format_bytes ------------------------------------------------------------


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        (None, "unknown"),
        (0, "0 B"),
        (1, "1 B"),
        (1023, "1023 B"),
        (1024, "1.0 KiB"),
        (1536, "1.5 KiB"),
        (1048576, "1.0 MiB"),
        (1610612736, "1.5 GiB"),
        (1099511627776, "1.0 TiB"),
    ],
)
def test_format_bytes(size: int | None, expected: str) -> None:
    assert format_bytes(size) == expected


def test_format_bytes_rejects_a_negative_size() -> None:
    with pytest.raises(ValueError, match="size"):
        format_bytes(-1)


# --- end-to-end against a real ingest ---------------------------------------


def test_a_real_batch_produces_the_same_counts_as_the_service() -> None:
    """Derivation, proven against actual output rather than a hand-built one."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "docs"
        root.mkdir()
        (root / "api.md").write_text("# API\nSome text.", encoding="utf-8")
        (root / "empty.txt").write_text("", encoding="utf-8")
        (root / "binary.txt").write_bytes(b"good\x00bad\x00bytes")
        (root / "logo.png").write_bytes(b"\x89PNG not really")

        result = ingest(root)
        report = build_report(result, duration_seconds=0.01)

    assert report.loaded == result.loaded_count
    assert report.skipped == result.skipped_count
    assert report.failed == result.failed_count
    assert report.unsupported == result.unsupported_count
    assert report.total_files == result.total_count
    assert report.indexable == result.loaded_count


def test_a_real_report_names_the_root_not_an_absolute_path() -> None:
    """A server path must not escape into a committed report."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "docs"
        root.mkdir()
        (root / "api.md").write_text("# API\n" + SECRET, encoding="utf-8")

        report = build_report(ingest(root))

    assert report.root == "docs"
    assert str(root) not in report.render()
    assert str(root) not in report.to_json()
    assert SECRET not in report.render()
    assert SECRET not in report.to_json()


def test_dedup_decisions_can_be_joined_to_a_real_batch() -> None:
    """The two modules compose without either knowing about the other's store."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "docs"
        root.mkdir()
        (root / "api.md").write_text("# API", encoding="utf-8")

        result = ingest(root)
        decisions = [
            classify([], source_path=f.relative_path, content=SECRET) for f in result.files
        ]
        report = build_report(result, decisions=decisions)

    assert dict(report.dedup)["new"] == 1  # type: ignore[arg-type]


# --- the index report --------------------------------------------------------


def index_outcome(
    path: str,
    action: PipelineAction,
    *,
    code: str | None = None,
    size: int | None = 100,
    chunk_count: int = 0,
    decision: DedupDecision | None = None,
    reason: str | None = None,
) -> IndexOutcome:
    """Build one synthetic pass outcome with only the fields a report reads.

    The file itself always *loaded*: a post-load failure (the provider, the
    registry) is exactly the case whose code exists nowhere but on the
    outcome, which is the property under test.
    """
    return IndexOutcome(
        file=file_result(path, FileOutcome.LOADED, size=size, reason=reason),
        doc_id="doc_" + "0" * 16,
        action=action,
        decision=decision,
        chunk_count=chunk_count,
        status=IngestStatus.INDEXED
        if action in (PipelineAction.CREATED, PipelineAction.REPLACED)
        else None,
        code=code,
        reason=reason,
    )


def index_batch(
    outcomes: list[IndexOutcome],
    *,
    root: str = "docs",
    unsupported: list[UnsupportedFile] | None = None,
) -> IndexResult:
    """Assemble synthetic outcomes into a pass result."""
    return IndexResult(
        root=root,
        outcomes=tuple(outcomes),
        unsupported=tuple(unsupported or ()),
    )


def test_index_counts_mirror_the_pass() -> None:
    result = index_batch(
        [
            index_outcome("a.md", PipelineAction.CREATED, chunk_count=3),
            index_outcome("b.md", PipelineAction.REPLACED, chunk_count=2),
            index_outcome("c.md", PipelineAction.SKIPPED, code="unchanged"),
            index_outcome("d.md", PipelineAction.FAILED, code="encoding_error"),
        ],
        unsupported=[UnsupportedFile("logo.png", "logo.png", ".png", "unsupported")],
    )

    report = build_index_report(result)

    assert report.total_files == 5
    assert report.processed_files == 4
    assert report.created == 1
    assert report.replaced == 1
    assert report.skipped == 1
    assert report.failed == 1
    assert report.unsupported == 1
    assert report.chunks_written == 5
    assert report.has_failures
    assert not report.is_empty


def test_a_pass_with_no_files_is_empty_and_not_failing() -> None:
    report = build_index_report(index_batch([]))

    assert report.is_empty
    assert not report.has_failures
    assert report.total_bytes is None


def test_a_post_load_failure_groups_under_the_pipelines_code() -> None:
    """The outage's code is on the outcome — the load result says ``None``.

    Bucketing by the load's code would file a provider outage under
    "unspecified", hiding the one thing a reader needs after a failed pass.
    """
    outcome = index_outcome("a.md", PipelineAction.FAILED, code="dependency_unavailable")

    report = build_index_report(index_batch([outcome]))

    assert outcome.file.code is None
    assert [t.code for t in report.failures] == ["dependency_unavailable"]
    assert report.failures[0].paths == ("a.md",)


def test_skips_are_bucketed_by_the_pass_codes() -> None:
    result = index_batch(
        [
            index_outcome("a.md", PipelineAction.SKIPPED, code="unchanged"),
            index_outcome("b.md", PipelineAction.SKIPPED, code="unchanged"),
            index_outcome("c.md", PipelineAction.SKIPPED, code="empty_file"),
        ]
    )

    report = build_index_report(result)

    assert [(t.code, t.count) for t in report.skips] == [
        ("unchanged", 2),
        ("empty_file", 1),
    ]
    assert report.failures == ()


def test_index_total_bytes_sums_the_sizes_learned() -> None:
    result = index_batch(
        [
            index_outcome("a.md", PipelineAction.CREATED, size=100),
            index_outcome("b.md", PipelineAction.FAILED, code="x", size=None),
        ]
    )

    assert build_index_report(result).total_bytes == 100


def test_dedup_counts_the_decisions_the_pass_made() -> None:
    report = build_index_report(
        index_batch(
            [
                index_outcome("a.md", PipelineAction.CREATED, decision=decision(ChangeKind.NEW)),
                index_outcome("b.md", PipelineAction.REPLACED, decision=decision(ChangeKind.NEW)),
                index_outcome(
                    "c.md",
                    PipelineAction.SKIPPED,
                    code="unchanged",
                    decision=decision(ChangeKind.UNCHANGED),
                ),
            ]
        )
    )

    assert dict(report.dedup) == {  # type: ignore[arg-type]
        "new": 2,
        "unchanged": 1,
        "modified": 0,
        "renamed": 0,
    }


def test_dedup_is_none_when_nothing_reached_classification() -> None:
    """Every file failed to load, so dedup never ran — not "all zeros"."""
    report = build_index_report(
        index_batch([index_outcome("a.md", PipelineAction.FAILED, code="encoding_error")])
    )

    assert report.dedup is None


def test_index_duration_is_carried_through_and_validated() -> None:
    report = build_index_report(index_batch([]), duration_seconds=0.5)
    assert report.duration_seconds == 0.5
    assert build_index_report(index_batch([])).duration_seconds is None

    with pytest.raises(ValueError, match="duration_seconds"):
        build_index_report(index_batch([]), duration_seconds=-1.0)


def test_index_to_dict_carries_the_counts_and_groupings() -> None:
    result = index_batch(
        [
            index_outcome("a.md", PipelineAction.CREATED, chunk_count=4),
            index_outcome("b.md", PipelineAction.FAILED, code="pdf_encrypted"),
        ],
        unsupported=[UnsupportedFile("logo.png", "logo.png", ".png", "unsupported")],
    )

    payload = build_index_report(result, duration_seconds=0.01).to_dict()

    assert payload["created"] == 1
    assert payload["failed"] == 1
    assert payload["chunks_written"] == 4
    assert payload["unsupported"] == 1
    assert payload["failures"] == [{"code": "pdf_encrypted", "count": 1, "paths": ["b.md"]}]
    assert "dedup" not in payload
    assert json.loads(build_index_report(result, duration_seconds=0.01).to_json()) == payload


def test_the_index_payload_contains_no_document_text() -> None:
    outcome = index_outcome("a.md", PipelineAction.FAILED, code="x", reason=SECRET)
    report = build_index_report(index_batch([outcome]))

    assert SECRET not in json.dumps(report.to_dict())
    assert SECRET not in report.to_json()
    assert SECRET not in report.render()


# --- index report rendering --------------------------------------------------


def test_index_render_starts_with_the_title_and_root() -> None:
    assert (
        build_index_report(index_batch([], root="docs")).render().startswith("Index report: docs")
    )


def test_index_render_says_no_files_found_for_an_empty_pass() -> None:
    rendered = build_index_report(index_batch([])).render()

    assert "no files found" in rendered
    assert "chunks written" not in rendered


def test_index_render_summarises_discovered_attempted_and_chunks() -> None:
    report = build_index_report(
        index_batch(
            [
                index_outcome("a.md", PipelineAction.CREATED, chunk_count=7),
                index_outcome("b.md", PipelineAction.SKIPPED, code="unchanged"),
            ]
        )
    )

    assert "2 discovered | 2 attempted | 7 chunks written" in report.render()


def test_index_render_lists_each_outcome_count() -> None:
    report = build_index_report(
        index_batch(
            [
                index_outcome("a.md", PipelineAction.CREATED, chunk_count=1),
                index_outcome("b.md", PipelineAction.REPLACED, chunk_count=1),
                index_outcome("c.md", PipelineAction.SKIPPED, code="unchanged"),
                index_outcome("d.md", PipelineAction.FAILED, code="encoding_error"),
            ],
            unsupported=[UnsupportedFile("logo.png", "logo.png", ".png", "unsupported")],
        )
    )

    rendered = report.render()

    assert "created      1" in rendered
    assert "replaced     1" in rendered
    assert "skipped      1" in rendered
    assert "failed       1" in rendered
    assert "unsupported  1" in rendered


def test_index_render_has_no_trailing_newline() -> None:
    assert not build_index_report(index_batch([])).render().endswith("\n")


def test_index_render_is_deterministic_for_the_same_pass() -> None:
    def same_pass() -> IndexResult:
        return index_batch(
            [
                index_outcome("b.md", PipelineAction.FAILED, code="beta"),
                index_outcome("a.md", PipelineAction.FAILED, code="alpha"),
            ]
        )

    assert build_index_report(same_pass()).render() == build_index_report(same_pass()).render()


def test_the_index_report_is_immutable() -> None:
    report = build_index_report(index_batch([]))

    with pytest.raises(dataclasses.FrozenInstanceError):
        report.created = 99  # type: ignore[misc]


# --- end-to-end against a real index pass ------------------------------------


def test_a_real_pass_produces_the_same_counts_as_the_result() -> None:
    """Derivation, proven against a real pass over a real registry."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "docs"
        root.mkdir()
        (root / "api.md").write_text("# API\nSome text.", encoding="utf-8")
        (root / "empty.txt").write_text("", encoding="utf-8")
        (root / "logo.png").write_bytes(b"\x89PNG not really")
        registry = DocumentRegistry(path=Path(tmp) / "registry.sqlite3")

        result = index_directory(
            root,
            embedder=FakeEmbedder(),
            store=FakeChunkStore(),
            registry=registry,
        )
        report = build_index_report(result, duration_seconds=0.01)
        registry.close()

    assert report.created == result.created_count
    assert report.skipped == result.skipped_count
    assert report.failed == result.failed_count
    assert report.unsupported == len(result.unsupported)
    assert report.total_files == result.total_count
    assert report.chunks_written == result.chunks_written


def test_a_real_index_report_names_the_root_not_the_absolute_path() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "docs"
        root.mkdir()
        (root / "api.md").write_text("# API\n" + SECRET, encoding="utf-8")
        registry = DocumentRegistry(path=Path(tmp) / "registry.sqlite3")

        report = build_index_report(
            index_directory(
                root, embedder=FakeEmbedder(), store=FakeChunkStore(), registry=registry
            )
        )
        registry.close()

    assert report.root == "docs"
    assert str(root) not in report.render()
    assert str(root) not in report.to_json()
    assert SECRET not in report.render()
    assert SECRET not in report.to_json()
