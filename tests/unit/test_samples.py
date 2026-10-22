"""Tests for the sample corpus generator (:mod:`scripts.samples`).

Two properties carry most of the weight here. The first is *determinism*:
ingestion identifies a document by hashing its path and its bytes, so a
generator that produced a slightly different file each run would make every
re-run of the pipeline report the whole corpus as modified. The second is
that the corpus genuinely exercises the pipeline, which is checked by
running the real ``ingest()`` over it rather than by asserting on the
generator's own description of itself.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.ingestion.pdf_loader import load_pdf
from app.ingestion.service import ingest
from scripts.samples import PrepareResult, prepare, sample_files


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """A freshly prepared sample corpus."""
    destination = tmp_path / "samples"
    prepare(destination)
    return destination


def _bytes_by_path(root: Path) -> dict[str, bytes]:
    """Every file under ``root``, keyed by its forward-slashed relative path."""
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


# ---------------------------------------------------------------------------
# The plan itself
# ---------------------------------------------------------------------------
def test_sample_files_is_stable_across_calls() -> None:
    """Two calls describe the same corpus, in the same order, byte for byte.

    ``sample_files()`` is what both ``prepare()`` and the tests read, so a
    change here is a change to the corpus itself and must be deliberate.
    """
    first = sample_files()
    second = sample_files()

    assert [f.relative_path for f in first] == [f.relative_path for f in second]
    assert [f.data for f in first] == [f.data for f in second]
    assert first == second


def test_relative_paths_are_portable() -> None:
    """Paths are relative, forward-slashed, and free of traversal.

    ``prepare()`` joins them onto a destination directory, so a backslash
    would build the wrong path on Linux and an absolute one would write
    outside the corpus entirely.
    """
    for planned in sample_files():
        assert not planned.relative_path.startswith("/")
        assert "\\" not in planned.relative_path
        assert ".." not in planned.relative_path.split("/")
        assert planned.path.is_absolute() is False


def test_readable_corpus_and_edge_cases_are_separate() -> None:
    """The documents worth retrieving sit apart from the ones meant to fail.

    A later evaluation stage wants a corpus of usable documents; wanting a
    demonstration of every failure code at the same time would force it to
    filter. Two directories let each caller choose.
    """
    areas = {f.relative_path.split("/", 1)[0] for f in sample_files()}

    assert areas == {"corpus", "edge-cases"}
    assert sum(1 for f in sample_files() if f.relative_path.startswith("corpus/")) == 5


def test_every_planned_file_says_what_it_demonstrates() -> None:
    """The ``note`` is what ``prepare()`` reports, so it must not be empty."""
    for planned in sample_files():
        assert planned.note.strip()


# ---------------------------------------------------------------------------
# Writing the corpus
# ---------------------------------------------------------------------------
def test_prepare_writes_every_planned_file(tmp_path: Path) -> None:
    """A first run creates the whole corpus and reports it as created."""
    destination = tmp_path / "fresh"

    result = prepare(destination)

    assert result.created == tuple(f.relative_path for f in sample_files())
    assert result.unchanged == ()
    assert result.updated == ()
    assert result.kept == ()
    assert result.total == len(sample_files())
    assert set(_bytes_by_path(destination)) == {f.relative_path for f in sample_files()}


def test_prepare_creates_missing_parent_directories(tmp_path: Path) -> None:
    """A destination several levels deep works without manual mkdir calls."""
    destination = tmp_path / "a" / "b" / "samples"

    prepare(destination)

    assert (destination / "corpus" / "api-spec.md").is_file()


def test_prepare_is_idempotent_over_an_untouched_corpus(tmp_path: Path) -> None:
    """A second run writes nothing and reports every file unchanged."""
    destination = tmp_path / "samples"
    prepare(destination)
    before = _bytes_by_path(destination)

    result = prepare(destination)

    assert result.created == ()
    assert result.updated == ()
    assert result.kept == ()
    assert result.unchanged == tuple(f.relative_path for f in sample_files())
    assert _bytes_by_path(destination) == before


def test_regeneration_produces_identical_bytes(tmp_path: Path) -> None:
    """Two independent runs write byte-identical files.

    This is the determinism claim in the module docstring, tested across
    directories rather than across time: if the builders were unstable, the
    two corpora would differ and every re-ingest would report *modified*.
    """
    first = tmp_path / "run-one"
    second = tmp_path / "run-two"
    prepare(first)
    prepare(second)

    assert _bytes_by_path(first) == _bytes_by_path(second)


def test_edited_file_is_kept_without_force(tmp_path: Path) -> None:
    """A file that differs is left exactly as the user wrote it."""
    destination = tmp_path / "samples"
    prepare(destination)
    edited = destination / "corpus" / "faq.txt"
    edited.write_text("hand written notes", encoding="utf-8")

    result = prepare(destination)

    assert result.created == ()
    assert result.updated == ()
    assert result.unchanged == tuple(
        f.relative_path for f in sample_files() if f.relative_path != "corpus/faq.txt"
    )
    assert result.kept == ("corpus/faq.txt",)
    assert edited.read_text(encoding="utf-8") == "hand written notes"


def test_force_replaces_an_edited_file(tmp_path: Path) -> None:
    """``force`` is the only thing that overwrites, and it overwrites exactly."""
    destination = tmp_path / "samples"
    prepare(destination)
    edited = destination / "corpus" / "faq.txt"
    edited.write_text("hand written notes", encoding="utf-8")
    planned = next(f for f in sample_files() if f.relative_path == "corpus/faq.txt")

    result = prepare(destination, force=True)

    assert result.updated == ("corpus/faq.txt",)
    assert result.created == ()
    assert result.kept == ()
    assert edited.read_bytes() == planned.data


def test_prepare_never_deletes_anything(tmp_path: Path) -> None:
    """A file the generator does not know about survives a run untouched.

    ``prepare()`` writes a corpus; it does not manage a directory. Removing
    stray files would silently destroy work the moment a name changed.
    """
    destination = tmp_path / "samples"
    prepare(destination)
    stray = destination / "corpus" / "my-own-notes.md"
    stray.write_text("mine", encoding="utf-8")

    prepare(destination, force=True)

    assert stray.read_text(encoding="utf-8") == "mine"
    assert len(_bytes_by_path(destination)) == len(sample_files()) + 1


# ---------------------------------------------------------------------------
# The bytes the edge cases claim to contain
# ---------------------------------------------------------------------------
def test_edge_case_files_have_the_bytes_they_advertise(tmp_path: Path) -> None:
    """Each edge case really is the kind of file it claims to be.

    A generator that wrote a merely-empty file while claiming a NUL-filled
    one would silently pass a loader test that never sees a failure.
    """
    destination = tmp_path / "samples"
    prepare(destination)

    assert (destination / "edge-cases" / "empty-notes.txt").stat().st_size == 0

    broken = (destination / "edge-cases" / "broken-export.txt").read_bytes()
    assert b"\x00" in broken

    logo = (destination / "edge-cases" / "logo.png").read_bytes()
    assert logo.startswith(b"\x89PNG\r\n\x1a\n")


def test_generated_pdfs_load_through_the_real_loader(tmp_path: Path) -> None:
    """The sample PDF has text to extract; the scan has none.

    Asserted against :func:`load_pdf` rather than by inspecting the PDF
    syntax, because the point of the samples is what the pipeline reads out
    of them, not how they are encoded.
    """
    destination = tmp_path / "samples"
    prepare(destination)

    architecture = load_pdf(destination / "corpus" / "architecture.pdf")
    assert architecture.page_count == 3
    assert architecture.is_empty is False
    assert "ledger" in architecture.text

    scan = load_pdf(destination / "edge-cases" / "scanned-contract.pdf")
    assert scan.is_empty is True


# ---------------------------------------------------------------------------
# The corpus as a whole
# ---------------------------------------------------------------------------
def test_prepared_corpus_exercises_every_outcome(tmp_path: Path) -> None:
    """The real pipeline reports every outcome category over the samples.

    This is the test that makes the corpus worth having: one directory that
    produces loaded, skipped, failed *and* unsupported results, each with a
    stable code, so a later stage can be demonstrated end to end.
    """
    destination = tmp_path / "samples"
    prepare(destination)

    result = ingest(destination)

    assert result.total_count == 10
    assert result.loaded_count == 5
    assert result.skipped_count == 2
    assert result.failed_count == 2
    assert result.unsupported_count == 1
    assert {f.code for f in result.skipped()} == {"empty_file", "no_text_layer"}
    assert {f.code for f in result.failed()} == {"binary_content", "pdf_encrypted"}


def test_prepared_corpus_is_recognised_as_unchanged_on_reingest(tmp_path: Path) -> None:
    """Ingesting twice reports the same paths, so nothing looks modified.

    The report carries the paths discovery found; the hashing that would
    classify a change happens at the storage stage, which is not built yet.
    This pins the half of the determinism story available today: the same
    inputs are handed over on both runs.
    """
    destination = tmp_path / "samples"
    prepare(destination)

    first = ingest(destination)
    second = ingest(destination)

    assert [f.relative_path for f in first.files] == [f.relative_path for f in second.files]
    assert [f.byte_size for f in first.files] == [f.byte_size for f in second.files]


# ---------------------------------------------------------------------------
# The result object
# ---------------------------------------------------------------------------
def test_summary_is_json_safe_and_complete(tmp_path: Path) -> None:
    """``summary()`` is what the CLI prints as JSON, so it must serialise."""
    destination = tmp_path / "samples"
    (destination / "corpus").mkdir(parents=True)
    (destination / "corpus" / "api-spec.md").write_text("edited", encoding="utf-8")
    result = prepare(destination)

    summary = json.loads(json.dumps(result.summary()))

    assert summary["total"] == len(sample_files())
    assert len(summary["created"]) == len(sample_files()) - 1
    assert summary["unchanged"] == []
    assert summary["kept"] == ["corpus/api-spec.md"]
    assert summary["destination"].endswith("samples")


def test_render_marks_each_file_and_explains_what_kept_means(tmp_path: Path) -> None:
    """The text form names every file, and says how to resolve a held one."""
    destination = tmp_path / "samples"
    prepare(destination)
    (destination / "corpus" / "faq.txt").write_text("edited", encoding="utf-8")

    result = prepare(destination)
    rendered = result.render()

    for planned in sample_files():
        assert planned.relative_path in rendered
    assert "! kept" in rendered
    assert "--force" in rendered


def test_render_of_a_clean_run_has_no_warnings(tmp_path: Path) -> None:
    """A corpus that is already correct renders without anything to act on."""
    destination = tmp_path / "samples"
    prepare(destination)

    rendered = prepare(destination).render()

    assert "= unchanged" in rendered
    assert "!" not in rendered
    assert "already differ" not in rendered


def test_result_is_immutable() -> None:
    """Reports handed to a renderer cannot be rewritten underneath it."""
    result = PrepareResult(
        destination=Path("somewhere"),
        created=("a.txt",),
        unchanged=(),
        updated=(),
        kept=(),
    )

    with pytest.raises((AttributeError, TypeError)):
        result.created = ()  # type: ignore[misc]
