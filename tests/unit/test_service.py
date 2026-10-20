"""Tests for the ingest service (discovery -> load -> metadata).

The service is the first component that has to be trustworthy under a
*mixture* of outcomes: a real directory contains good files, blank files,
scans, corrupt files and things we do not handle, all at once. So most of
these tests build a deliberately awkward tree rather than a single fixture,
and then assert that every file came out with the outcome it deserves.

Two behaviours get particular attention:

* **No single file may abort the batch.** A missing file, a corrupt PDF and
  a rejected password all return a ``failed`` result so the rest of the
  directory still gets processed.
* **Nothing derived may raise for the caller's own directory.** Discovery's
  containment errors are the documented exceptions, because those are the
  *caller* passing a bad root rather than a bad file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.domain.errors import NotFoundError, ValidationError
from app.ingestion.service import (
    EMPTY_FILE_CODE,
    LOADABLE_MEDIA_TYPES,
    NO_TEXT_LAYER_CODE,
    FileOutcome,
    FileResult,
    IngestResult,
    describe,
    ingest,
    ingest_file,
)
from tests.unit.pdf_fixtures import (
    build_blank_pdf,
    build_encrypted_pdf,
    build_pdf,
)


def write(path: Path, data: str | bytes) -> Path:
    """Create ``path``'s parent if needed and write ``data`` as given."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, bytes):
        path.write_bytes(data)
    else:
        path.write_text(data, encoding="utf-8")
    return path


def mixed_tree(root: Path) -> Path:
    """Build the awkward directory most of these tests run against.

    Deliberately contains one of everything that matters: two good files, a
    nested one, an empty text file, a scanned PDF, a corrupt PDF, an
    unsupported extension and a hidden file that discovery must ignore.
    """
    write(root / "spec.md", "# Payment Gateway\n\nReal body text here.\n")
    write(root / "notes.txt", "Just some plain notes.\n")
    write(root / "sub" / "nested.md", "## Nested heading\n\nnested body\n")
    write(root / "empty.txt", "")
    write(root / "scan.pdf", build_blank_pdf(2))
    write(root / "broken.pdf", b"%PDF-1.4\nthis is not really a pdf")
    write(root / "image.png", b"\x89PNG\r\n\x1a\n")
    write(root / ".hidden.md", "should be ignored\n")
    return root


# --- single-file loading ---------------------------------------------------


def test_markdown_file_is_loaded_with_metadata(tmp_path) -> None:
    path = write(tmp_path / "spec.md", "# Payment Gateway\n\nReal body text here.\n")
    result = ingest_file(path)

    assert result.is_loaded
    assert result.media_type == "text/markdown"
    assert result.relative_path == "spec.md"
    assert result.byte_size == path.stat().st_size
    assert result.document is not None
    assert result.metadata is not None
    assert result.metadata.title == "Payment Gateway"
    assert result.metadata.word_count == 7


def test_plain_text_file_is_loaded(tmp_path) -> None:
    result = ingest_file(write(tmp_path / "notes.txt", "plain notes\n"))

    assert result.is_loaded
    assert result.media_type == "text/plain"
    assert result.page_count is None  # plain text has no pages


def test_pdf_is_loaded_with_a_page_count(tmp_path) -> None:
    path = write(tmp_path / "doc.pdf", build_pdf(["page one", "page two"]))
    result = ingest_file(path)

    assert result.is_loaded
    assert result.media_type == "application/pdf"
    assert result.page_count == 2
    assert result.metadata is not None
    assert result.metadata.page_count == 2


def test_pdf_suffix_resolves_to_the_pdf_loader(tmp_path) -> None:
    """Regression guard: a direct PDF call must return a result, not raise.

    ``media_type_for_path`` raises for ``.pdf`` — it belongs to the text
    loader module, which cannot read PDFs and says so. The service *can*
    read both, so it resolves the suffix itself; without that, calling
    ``ingest_file`` on a PDF would throw out of the function instead of
    returning a result, breaking the documented contract that only bad
    caller arguments raise.
    """
    path = write(tmp_path / "doc.pdf", build_pdf(["hello"]))
    result = ingest_file(path)

    assert result.is_loaded
    assert result.media_type == "application/pdf"
    assert result.page_count == 1


def test_explicit_media_type_override_is_used(tmp_path) -> None:
    path = write(tmp_path / "doc.pdf", build_pdf(["hi"]))
    result = ingest_file(path, media_type="application/pdf")

    assert result.is_loaded
    assert result.media_type == "application/pdf"


def test_relative_path_defaults_to_the_basename(tmp_path) -> None:
    assert ingest_file(write(tmp_path / "a.txt", "hi")).relative_path == "a.txt"


def test_relative_path_override_records_the_scanned_location(tmp_path) -> None:
    path = write(tmp_path / "a.txt", "hi")
    result = ingest_file(path, relative_path="uploads/nested/a.txt")

    assert result.relative_path == "uploads/nested/a.txt"
    assert result.source_name == "a.txt"  # basename is still just the name


def test_text_property_exposes_the_document_text(tmp_path) -> None:
    result = ingest_file(write(tmp_path / "a.txt", "hello world"))

    assert result.text == "hello world"


def test_loaded_results_carry_no_skip_or_error_code(tmp_path) -> None:
    result = ingest_file(write(tmp_path / "a.txt", "hello"))

    assert result.code is None
    assert result.reason is None


def test_loadable_media_types_cover_every_supported_format(tmp_path) -> None:
    assert LOADABLE_MEDIA_TYPES == {"text/plain", "text/markdown", "application/pdf"}


# --- skipping: valid files with nothing to index ---------------------------


def test_empty_text_file_is_skipped_not_failed(tmp_path) -> None:
    result = ingest_file(write(tmp_path / "empty.txt", ""))

    assert result.is_skipped
    assert not result.is_failed
    assert result.code == EMPTY_FILE_CODE
    assert "nothing to index" in (result.reason or "")


def test_whitespace_only_file_is_skipped(tmp_path) -> None:
    """Whitespace is not text, so it has nothing to index either."""
    result = ingest_file(write(tmp_path / "blank.txt", "   \n\t\n"))

    assert result.is_skipped
    assert result.code == EMPTY_FILE_CODE


def test_scanned_pdf_is_skipped_with_a_distinct_code(tmp_path) -> None:
    result = ingest_file(write(tmp_path / "scan.pdf", build_blank_pdf(2)))

    assert result.is_skipped
    assert result.code == NO_TEXT_LAYER_CODE
    assert "no text layer" in (result.reason or "")
    assert result.page_count == 2


def test_skip_codes_differ_between_a_blank_file_and_a_scan(tmp_path) -> None:
    """The two cases call for different user actions, so they stay distinct."""
    blank = ingest_file(write(tmp_path / "a.txt", ""))
    scan = ingest_file(write(tmp_path / "b.pdf", build_blank_pdf(1)))

    assert blank.code != scan.code


def test_skipped_file_still_has_its_document(tmp_path) -> None:
    """It loaded fine — it simply had no text — so both are present."""
    result = ingest_file(write(tmp_path / "scan.pdf", build_blank_pdf(1)))

    assert result.document is not None
    assert result.metadata is not None
    # The text property is a proxy for the document's own text, and a scan
    # has no usable content in it whatever the page joiner produces.
    assert result.text == result.document.text
    assert not result.text.strip()


def test_skipped_file_still_reports_provenance_metadata(tmp_path) -> None:
    result = ingest_file(write(tmp_path / "empty.txt", ""))

    assert result.metadata is not None
    assert result.metadata.source_name == "empty.txt"
    assert result.metadata.word_count == 0
    # The filename-derived title is recorded as such, not passed off real
    assert str(result.metadata.title_source) == "filename"


# --- failures: no single file may abort the batch --------------------------


def test_missing_file_returns_a_failed_result(tmp_path) -> None:
    result = ingest_file(tmp_path / "does-not-exist.txt")

    assert result.is_failed
    assert result.code == "file_not_found"
    assert result.reason
    assert result.document is None
    assert result.metadata is None


def test_corrupt_pdf_is_failed_with_invalid_pdf(tmp_path) -> None:
    result = ingest_file(write(tmp_path / "bad.pdf", b"%PDF-1.4\nnonsense"))

    assert result.is_failed
    assert result.code == "invalid_pdf"


def test_encrypted_pdf_without_a_password_is_failed(tmp_path) -> None:
    path = write(tmp_path / "locked.pdf", build_encrypted_pdf(["secret"], "pw123"))
    result = ingest_file(path)

    assert result.is_failed
    assert result.code == "pdf_encrypted"


def test_non_utf8_file_is_failed_with_encoding_error(tmp_path) -> None:
    result = ingest_file(write(tmp_path / "a.txt", b"\xff\xfe not text"))

    assert result.is_failed
    assert result.code == "encoding_error"


def test_nul_bytes_in_a_text_file_are_failed_as_binary(tmp_path) -> None:
    result = ingest_file(write(tmp_path / "a.txt", b"looks like\0 a docx"))

    assert result.is_failed
    assert result.code == "binary_content"


def test_failed_result_has_no_size_when_none_was_supplied(tmp_path) -> None:
    """Honest about not knowing: the size is never read for a missing file."""
    result = ingest_file(tmp_path / "gone.txt")

    assert result.byte_size is None


def test_failed_result_uses_a_callers_supplied_size(tmp_path) -> None:
    """Discovery already stat'ed the file, so its size is available to us."""
    result = ingest_file(tmp_path / "gone.txt", size_bytes=4096)

    assert result.byte_size == 4096


def test_failure_reason_is_the_errors_own_message(tmp_path) -> None:
    result = ingest_file(tmp_path / "gone.txt")

    assert result.reason == "file does not exist: gone.txt"


# --- caller argument errors are raised, not swallowed ----------------------


def test_unloadable_caller_supplied_media_type_raises(tmp_path) -> None:
    """The caller *stated* this type, so getting it wrong is their bug.

    Reporting it as a failed file instead would let a programming mistake
    ship as a per-file data problem.
    """
    path = write(tmp_path / "a.txt", "hi")

    with pytest.raises(ValidationError) as excinfo:
        ingest_file(path, media_type="application/x-nonsense")

    assert excinfo.value.code == "unsupported_media_type"


def test_the_raised_message_lists_what_is_supported(tmp_path) -> None:
    with pytest.raises(ValidationError) as excinfo:
        ingest_file(write(tmp_path / "a.txt", "hi"), media_type="application/x-nonsense")

    assert "text/markdown" in str(excinfo.value)
    assert "application/pdf" in str(excinfo.value)


def test_loadable_caller_supplied_media_type_does_not_raise(tmp_path) -> None:
    for media_type in LOADABLE_MEDIA_TYPES:
        path = (
            write(tmp_path / "d.pdf", build_pdf(["x"]))
            if media_type == "application/pdf"
            else write(tmp_path / "d.txt", "x")
        )
        assert ingest_file(path, media_type=media_type).is_loaded


def test_load_refuses_a_media_type_it_cannot_handle(tmp_path) -> None:
    """The helper guards its own contract, independent of its caller.

    ``ingest_file`` validates a caller-stated type first, so today this
    branch is not reachable through the public API — which is exactly why
    it is tested directly rather than left as unexecuted defensive code. It
    stays because the helper would otherwise fall off the end and return
    ``None``, and because a future resolver must not be able to smuggle an
    unloadable type past it.
    """
    from app.ingestion.service import _load

    with pytest.raises(ValidationError) as excinfo:
        _load(write(tmp_path / "a.txt", "hi"), "application/x-nonsense")

    assert excinfo.value.code == "unsupported_media_type"


# --- unsupported files -----------------------------------------------------


def test_unknown_extension_is_classified_by_discovery_not_the_service(tmp_path) -> None:
    """`.png` is unsupported *discovery's* call to make, before any reading."""
    mixed_tree(tmp_path)

    result = ingest(tmp_path)

    assert [u.relative_path for u in result.unsupported] == ["image.png"]
    assert "image.png" not in result.relative_paths()
    assert "extension '.png' is not supported" in result.unsupported[0].reason


def test_unsupported_files_never_count_as_failures(tmp_path) -> None:
    mixed_tree(tmp_path)

    result = ingest(tmp_path)

    # image.png is reported by discovery, so it is not in files at all and
    # cannot appear among the failures -- the only failure is the corrupt PDF.
    assert result.unsupported_count == 1
    assert [f.source_name for f in result.failed()] == ["broken.pdf"]
    assert all(f.source_name != "image.png" for f in result.files)


def test_unsupported_files_are_carried_through_with_their_reason(tmp_path) -> None:
    mixed_tree(tmp_path)

    result = ingest(tmp_path)

    assert result.unsupported[0].filename == "image.png"
    assert result.unsupported[0].suffix == ".png"


# --- orchestrating a whole directory ---------------------------------------


def test_mixed_directory_produces_the_expected_counts(tmp_path) -> None:
    mixed_tree(tmp_path)

    result = ingest(tmp_path)

    assert result.loaded_count == 3  # spec.md, notes.txt, sub/nested.md
    assert result.skipped_count == 2  # empty.txt, scan.pdf
    assert result.failed_count == 1  # broken.pdf
    assert result.unsupported_count == 1  # image.png
    assert result.processed_count == 6
    assert result.total_count == 7


def test_nested_files_are_found_and_recursed_into(tmp_path) -> None:
    mixed_tree(tmp_path)

    assert "sub/nested.md" in ingest(tmp_path).relative_paths()


def test_relative_paths_use_forward_slashes(tmp_path) -> None:
    """make_doc_id hashes this string later, so it must match across OSes."""
    mixed_tree(tmp_path)

    paths = ingest(tmp_path).relative_paths()

    assert "\\" not in " ".join(paths)


def test_non_recursive_scan_stays_at_the_top_level(tmp_path) -> None:
    mixed_tree(tmp_path)

    paths = ingest(tmp_path, recursive=False).relative_paths()

    assert "sub/nested.md" not in paths
    assert "spec.md" in paths


def test_hidden_files_are_not_attempted(tmp_path) -> None:
    mixed_tree(tmp_path)

    assert ".hidden.md" not in ingest(tmp_path).relative_paths()


def test_results_are_in_a_stable_order_across_runs(tmp_path) -> None:
    """The same directory must produce the same order every time.

    Determinism is the guarantee worth asserting here — the exact ordering
    belongs to discovery, which owns the walk.
    """
    mixed_tree(tmp_path)

    assert ingest(tmp_path).relative_paths() == ingest(tmp_path).relative_paths()


def test_one_bad_file_does_not_stop_the_good_ones(tmp_path) -> None:
    """The central batch guarantee: failures are recorded, not propagated."""
    write(tmp_path / "aaa_broken.pdf", b"garbage")
    write(tmp_path / "zzz_good.txt", "fine\n")

    result = ingest(tmp_path)

    assert result.has_failures
    assert [f.relative_path for f in result.failed()] == ["aaa_broken.pdf"]
    assert [f.relative_path for f in result.files if f.is_loaded] == ["zzz_good.txt"]


def test_a_batch_where_every_file_failed_still_returns(tmp_path) -> None:
    """All-failures is information, not an exception."""
    write(tmp_path / "a.txt", b"\xff\xfe")
    write(tmp_path / "b.txt", b"\xff\xfe")

    result = ingest(tmp_path)

    assert result.loaded_count == 0
    assert result.failed_count == 2
    assert result.has_failures
    assert not result.is_empty


def test_root_is_recorded_as_a_name_never_a_path(tmp_path) -> None:
    """An absolute server path must not escape into a log or a response."""
    mixed_tree(tmp_path)

    root_name = ingest(tmp_path).root

    assert root_name == tmp_path.name
    assert str(tmp_path) not in root_name


def test_an_empty_directory_is_a_result_not_an_error(tmp_path) -> None:
    result = ingest(tmp_path)

    assert result.is_empty
    assert result.processed_count == 0
    assert result.files == ()
    assert not result.has_failures


def test_a_directory_holding_only_gitkeep_reads_as_empty(tmp_path) -> None:
    write(tmp_path / ".gitkeep", "")

    result = ingest(tmp_path)

    assert result.is_empty
    assert result.processed_count == 0


def test_missing_root_raises_not_found(tmp_path) -> None:
    with pytest.raises(NotFoundError) as excinfo:
        ingest(tmp_path / "no-such-dir")

    assert excinfo.value.code == "discovery_root_not_found"


def test_root_that_is_a_file_raises_validation_error(tmp_path) -> None:
    with pytest.raises(ValidationError) as excinfo:
        ingest(write(tmp_path / "a.txt", "hi"))

    assert excinfo.value.code == "discovery_root_not_a_directory"


def test_root_outside_the_base_dir_is_refused(tmp_path) -> None:
    """Containment still applies — the service never opens anything first."""
    inside = tmp_path / "inside"
    inside.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    write(outside / "a.txt", "hi")

    with pytest.raises(ValidationError) as excinfo:
        ingest(outside, base_dir=inside)

    assert excinfo.value.code == "discovery_root_outside_base"


def test_base_dir_that_contains_the_root_allows_the_scan(tmp_path) -> None:
    write(tmp_path / "inside" / "a.txt", "hi")

    result = ingest(tmp_path / "inside", base_dir=tmp_path)

    assert result.loaded_count == 1


# --- outcome flags ---------------------------------------------------------


def test_outcome_flags_are_mutually_exclusive(tmp_path) -> None:
    loaded = ingest_file(write(tmp_path / "a.txt", "hi"))
    skipped = ingest_file(write(tmp_path / "b.txt", ""))
    failed = ingest_file(tmp_path / "c.txt")

    for result in (loaded, skipped, failed):
        flags = (result.is_loaded, result.is_skipped, result.is_failed)
        assert sum(flags) == 1, result.outcome


def test_is_processed_covers_loaded_and_skipped_but_not_failed(tmp_path) -> None:
    assert ingest_file(write(tmp_path / "a.txt", "hi")).is_processed
    assert ingest_file(write(tmp_path / "b.txt", "")).is_processed
    assert not ingest_file(tmp_path / "c.txt").is_processed


def test_text_is_empty_when_there_is_no_document(tmp_path) -> None:
    assert ingest_file(tmp_path / "gone.txt").text == ""


def test_file_outcome_is_a_string_enum(tmp_path) -> None:
    """So it serialises straight to JSON without a custom encoder."""
    result = ingest_file(write(tmp_path / "a.txt", "hi"))

    assert json.loads(json.dumps({"outcome": result.outcome})) == {"outcome": "loaded"}
    assert result.outcome == "loaded"
    assert {o.value for o in FileOutcome} == {"loaded", "skipped", "failed"}


# --- counts and accessors --------------------------------------------------


def test_counts_add_up_to_the_processed_total(tmp_path) -> None:
    mixed_tree(tmp_path)

    result = ingest(tmp_path)

    assert (
        result.loaded_count + result.skipped_count + result.failed_count == result.processed_count
    )
    assert result.processed_count + result.unsupported_count == result.total_count


def test_indexable_count_aliases_loaded_count(tmp_path) -> None:
    mixed_tree(tmp_path)

    result = ingest(tmp_path)

    assert result.indexable_count == result.loaded_count


def test_failed_and_skipped_selectors_return_only_their_own_kind(tmp_path) -> None:
    mixed_tree(tmp_path)

    result = ingest(tmp_path)

    assert all(f.is_failed for f in result.failed())
    assert all(f.is_skipped for f in result.skipped())
    assert [f.source_name for f in result.skipped()] == ["empty.txt", "scan.pdf"]


def test_has_failures_is_false_for_a_clean_batch(tmp_path) -> None:
    write(tmp_path / "a.txt", "hi")
    write(tmp_path / "b.txt", "")

    assert not ingest(tmp_path).has_failures


# --- summaries: safe for logs ----------------------------------------------


def test_file_result_summary_never_contains_document_text(tmp_path) -> None:
    body = "the confidential contract clause nine"
    result = ingest_file(write(tmp_path / "a.txt", f"# Title\n\n{body}"))

    assert body not in json.dumps(result.summary())


def test_file_result_summary_reports_the_outcome_and_provenance(tmp_path) -> None:
    result = ingest_file(write(tmp_path / "a.txt", "hello"))
    summary = result.summary()

    assert summary["outcome"] == "loaded"
    assert summary["source_name"] == "a.txt"
    assert summary["media_type"] == "text/plain"
    assert summary["word_count"] == 1


def test_file_result_summary_omits_absent_fields(tmp_path) -> None:
    """No code or reason for a success, and no page count for plain text."""
    summary = ingest_file(write(tmp_path / "a.txt", "hello")).summary()

    assert "code" not in summary
    assert "reason" not in summary
    assert "page_count" not in summary


def test_file_result_summary_includes_the_failure_details(tmp_path) -> None:
    summary = ingest_file(tmp_path / "gone.txt").summary()

    assert summary["code"] == "file_not_found"
    assert summary["outcome"] == "failed"


def test_file_result_summary_reports_a_pdfs_page_count(tmp_path) -> None:
    """Page count is the one field that only a paged format ever supplies."""
    path = write(tmp_path / "doc.pdf", build_pdf(["page one", "page two"]))

    assert ingest_file(path).summary()["page_count"] == 2


def test_file_result_summary_omits_the_title_when_metadata_is_absent(tmp_path) -> None:
    """A failed load built no metadata, so there is no title to report."""
    summary = ingest_file(tmp_path / "gone.txt").summary()

    assert "title" not in summary
    assert "word_count" not in summary


def test_ingest_result_summary_has_counts_and_problem_lists(tmp_path) -> None:
    mixed_tree(tmp_path)

    summary = ingest(tmp_path).summary()

    assert summary["root"] == tmp_path.name
    assert summary["loaded"] == 3
    assert summary["skipped"] == 2
    assert summary["failed"] == 1
    assert summary["unsupported"] == 1
    assert summary["failures"] == [{"path": "broken.pdf", "code": "invalid_pdf"}]
    assert [e["path"] for e in summary["skipped_files"]] == ["empty.txt", "scan.pdf"]


def test_summary_contains_no_absolute_paths(tmp_path) -> None:
    mixed_tree(tmp_path)

    payload = json.dumps(ingest(tmp_path).summary())

    assert str(tmp_path) not in payload


def test_summary_contains_no_document_text(tmp_path) -> None:
    body = "the confidential contract clause nine"
    write(tmp_path / "a.txt", f"## Title\n\n{body}")

    payload = json.dumps(ingest(tmp_path).summary())

    assert body not in payload


def test_summary_is_json_serialisable(tmp_path) -> None:
    """It is destined for structured logs and API responses."""
    mixed_tree(tmp_path)

    json.dumps(ingest(tmp_path).summary())  # must not raise


def test_describe_matches_the_summary_method(tmp_path) -> None:
    """Kept in symmetry with discovery's describe() so log calls stay uniform."""
    mixed_tree(tmp_path)

    result = ingest(tmp_path)

    assert describe(result) == result.summary()


# --- structure and exports -------------------------------------------------


def test_instantiating_an_empty_result_is_valid() -> None:
    """The dataclass is constructible on its own for tests and fakes."""
    empty = IngestResult(root="nowhere", files=(), unsupported=())

    assert empty.is_empty
    assert empty.total_count == 0


def test_module_exports_the_documented_names() -> None:
    import app.ingestion.service as service_module

    for name in (
        "EMPTY_FILE_CODE",
        "LOADABLE_MEDIA_TYPES",
        "NO_TEXT_LAYER_CODE",
        "FileOutcome",
        "FileResult",
        "IngestResult",
        "describe",
        "ingest",
        "ingest_file",
    ):
        assert hasattr(service_module, name), name


def test_file_result_is_immutable() -> None:
    """Frozen, so a report cannot have its results rewritten underneath it."""
    import dataclasses

    result = FileResult(
        source_name="a.txt",
        relative_path="a.txt",
        media_type="text/plain",
        byte_size=1,
        outcome=FileOutcome.LOADED,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.outcome = FileOutcome.FAILED  # type: ignore[misc]
