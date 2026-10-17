"""Tests for file discovery (Stage 10).

Organised around the three promises of the module:

1. **Classification** — supported vs unsupported, with actionable reasons.
2. **Containment** — nothing outside the root is ever returned. This is the
   security-critical property, so it gets dedicated symlink and ``..`` tests.
3. **Robustness** — empty directories, missing roots, hidden files, nested
   layouts and non-recursive scans behave predictably.

Everything runs against ``tmp_path``; no real project data is touched.
"""

from __future__ import annotations

import os

import pytest

from app.domain.errors import NotFoundError, ValidationError
from app.ingestion.discovery import (
    IGNORED_DIRECTORY_NAMES,
    DiscoveredFile,
    DiscoveryResult,
    UnsupportedFile,
    describe,
    discover,
)


def touch(path, content: str = "x") -> None:
    """Create a file (and its parents) with the given text."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# =============================================================================
# 1. Classification
# =============================================================================


def test_finds_each_supported_extension(tmp_path) -> None:
    touch(tmp_path / "a.pdf")
    touch(tmp_path / "b.txt")
    touch(tmp_path / "c.md")

    result = discover(tmp_path)

    assert result.relative_paths() == ("a.pdf", "b.txt", "c.md")
    assert {f.media_type for f in result.files} == {
        "application/pdf",
        "text/plain",
        "text/markdown",
    }


def test_extension_matching_is_case_insensitive(tmp_path) -> None:
    """``NOTES.MD`` must not be reported as unsupported on a case-sensitive scan."""
    touch(tmp_path / "NOTES.MD")
    touch(tmp_path / "Scan.PDF")

    result = discover(tmp_path)

    assert result.supported_count == 2
    assert result.unsupported_count == 0


def test_unsupported_file_reports_actionable_reason(tmp_path) -> None:
    touch(tmp_path / "report.docx")

    result = discover(tmp_path)

    assert result.supported_count == 0
    item = result.unsupported[0]
    assert isinstance(item, UnsupportedFile)
    assert item.suffix == ".docx"
    assert "'.docx'" in item.reason
    assert ".pdf" in item.reason and ".md" in item.reason  # lists what IS supported


def test_file_without_extension_is_reported(tmp_path) -> None:
    touch(tmp_path / "README")

    result = discover(tmp_path)

    assert result.unsupported_count == 1
    assert result.unsupported[0].suffix == ""
    assert "no extension" in result.unsupported[0].reason


def test_discovered_file_exposes_size(tmp_path) -> None:
    touch(tmp_path / "a.txt", "hello world")

    result = discover(tmp_path)

    assert result.files[0].size_bytes == len("hello world")
    assert result.total_size_bytes == len("hello world")


def test_discovered_file_suffix_property(tmp_path) -> None:
    touch(tmp_path / "nested" / "doc.MD")

    result = discover(tmp_path)

    assert result.files[0].suffix == ".md"
    assert result.files[0].relative_path == "nested/doc.MD"


def test_results_are_sorted_for_reproducibility(tmp_path) -> None:
    """Scan order must be stable so re-ingestion produces the same ids."""
    for name in ["zebra.md", "alpha.md", "middle.txt"]:
        touch(tmp_path / name)

    first = discover(tmp_path).relative_paths()
    second = discover(tmp_path).relative_paths()

    assert first == second
    assert first == ("alpha.md", "middle.txt", "zebra.md")


# =============================================================================
# 2. Containment (security)
# =============================================================================


def test_root_must_exist(tmp_path) -> None:
    with pytest.raises(NotFoundError) as excinfo:
        discover(tmp_path / "does-not-exist")

    assert excinfo.value.code == "discovery_root_not_found"
    # The client-facing message must not expose an absolute server path.
    assert str(tmp_path) not in excinfo.value.message
    assert "does-not-exist" in excinfo.value.message


def test_root_must_be_a_directory(tmp_path) -> None:
    file_path = tmp_path / "not-a-dir.txt"
    touch(file_path)

    with pytest.raises(ValidationError) as excinfo:
        discover(file_path)

    assert excinfo.value.code == "discovery_root_not_a_directory"


def test_root_outside_base_dir_is_rejected(tmp_path) -> None:
    """The configured data directory acts as a hard boundary."""
    inside = tmp_path / "data"
    inside.mkdir()
    outside = tmp_path / "elsewhere"
    outside.mkdir()

    with pytest.raises(ValidationError) as excinfo:
        discover(outside, base_dir=inside)

    assert excinfo.value.code == "discovery_root_outside_base"


def test_root_inside_base_dir_is_accepted(tmp_path) -> None:
    base = tmp_path / "data"
    (base / "docs").mkdir(parents=True)
    touch(base / "docs" / "a.md")

    result = discover(base / "docs", base_dir=base)

    assert result.relative_paths() == ("a.md",)


def test_base_dir_equal_to_root_is_accepted(tmp_path) -> None:
    touch(tmp_path / "a.md")

    assert discover(tmp_path, base_dir=tmp_path).supported_count == 1


def test_dot_dot_in_root_path_is_resolved_not_traversed(tmp_path) -> None:
    """``data/../secrets`` resolves before containment is evaluated."""
    (tmp_path / "data").mkdir()
    touch(tmp_path / "secrets" / "key.md", "secret")

    sneaky = tmp_path / "data" / ".." / "secrets"

    with pytest.raises(ValidationError) as excinfo:
        discover(sneaky, base_dir=tmp_path / "data")

    assert excinfo.value.code == "discovery_root_outside_base"


def test_relative_paths_never_contain_dot_dot(tmp_path) -> None:
    (tmp_path / "sub").mkdir()
    touch(tmp_path / "sub" / "a.md")

    result = discover(tmp_path)

    for rel in result.relative_paths():
        assert ".." not in rel
        assert not rel.startswith("/")


def test_relative_paths_use_forward_slashes(tmp_path) -> None:
    """Stable ids across platforms: make_doc_id hashes this string."""
    touch(tmp_path / "deep" / "nested" / "a.md")

    result = discover(tmp_path)

    # Note the trailing comma: without it this would be a bare string.
    assert result.relative_paths() == ("deep/nested/a.md",)


def _symlink_or_skip(link, target) -> None:
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError) as exc:  # pragma: no cover
        pytest.skip(f"symlinks unavailable on this host: {exc}")


def test_symlinked_file_pointing_outside_root_is_not_returned(tmp_path):
    """A link into /etc must never be indexed — the core containment promise."""
    root = tmp_path / "data"
    root.mkdir()
    outside = tmp_path / "outside-secret.md"
    touch(outside, "TOP SECRET")
    _symlink_or_skip(root / "link.md", outside)

    result = discover(root)

    assert result.relative_paths() == (), "a symlink escaping the root was returned as a document"
    assert result.unsupported_count == 1
    assert "outside" in result.unsupported[0].reason


def test_symlinked_directory_pointing_outside_root_is_not_descended(
    tmp_path,
) -> None:
    """Directory links are pruned, so C:\\Users can't be walked and indexed."""
    root = tmp_path / "data"
    root.mkdir()
    outside_dir = tmp_path / "outside_dir"
    touch(outside_dir / "secret.md", "TOP SECRET")
    _symlink_or_skip(root / "escape", outside_dir)

    result = discover(root)

    assert result.relative_paths() == ()
    assert result.is_empty is True


def test_symlink_target_inside_root_is_accepted(tmp_path) -> None:
    """Containment forbids escapes, not legitimate internal links."""
    root = tmp_path / "data"
    root.mkdir()
    real = root / "real.md"
    touch(real, "hello")
    _symlink_or_skip(root / "alias.md", real)

    result = discover(root)

    assert set(result.relative_paths()) == {"real.md", "alias.md"}


# =============================================================================
# 3. Robustness
# =============================================================================


def test_empty_directory_yields_empty_result_not_error(tmp_path) -> None:
    result = discover(tmp_path)

    assert isinstance(result, DiscoveryResult)
    assert result.is_empty is True
    assert result.files == ()
    assert result.unsupported == ()


def test_directory_with_only_hidden_files_counts_as_empty(tmp_path) -> None:
    """.gitkeep must not be mistaken for an unsupported document."""
    touch(tmp_path / ".gitkeep")

    result = discover(tmp_path)

    assert result.is_empty is True
    assert result.unsupported_count == 0


def test_hidden_files_are_skipped(tmp_path) -> None:
    touch(tmp_path / "real.md")
    touch(tmp_path / ".DS_Store")
    touch(tmp_path / ".hidden.md")

    result = discover(tmp_path)

    assert result.relative_paths() == ("real.md",)
    assert result.unsupported_count == 0


def test_ignored_directories_are_not_descended(tmp_path) -> None:
    """Walking .git or __pycache__ on every run would be slow and noisy."""
    touch(tmp_path / "real.md")
    for name in IGNORED_DIRECTORY_NAMES:
        touch(tmp_path / name / "junk.md")

    result = discover(tmp_path)

    assert result.relative_paths() == ("real.md",)


def test_nested_subdirectories_are_discovered_recursively(tmp_path) -> None:
    touch(tmp_path / "top.md")
    touch(tmp_path / "a" / "b" / "c" / "deep.txt")
    touch(tmp_path / "other" / "mid.pdf")

    result = discover(tmp_path)

    assert set(result.relative_paths()) == {
        "top.md",
        "a/b/c/deep.txt",
        "other/mid.pdf",
    }


def test_non_recursive_scan_returns_only_top_level(tmp_path) -> None:
    touch(tmp_path / "top.md")
    touch(tmp_path / "a" / "nested.md")

    result = discover(tmp_path, recursive=False)

    assert result.relative_paths() == ("top.md",)


def test_non_recursive_scan_still_prunes_ignored_dirs(tmp_path) -> None:
    touch(tmp_path / "top.md")

    result = discover(tmp_path, recursive=False)

    assert result.relative_paths() == ("top.md",)


def test_empty_subdirectory_does_not_break_discovery(tmp_path) -> None:
    (tmp_path / "empty-dir").mkdir()
    touch(tmp_path / "a.md")

    result = discover(tmp_path)

    assert result.relative_paths() == ("a.md",)


def test_directories_named_like_documents_are_not_treated_as_files(tmp_path) -> None:
    """A folder called ``notes.md`` must not be returned as a document."""
    (tmp_path / "notes.md").mkdir()

    result = discover(tmp_path)

    assert result.files == ()


def test_mixed_content_reports_both_lists(tmp_path) -> None:
    touch(tmp_path / "a.md")
    touch(tmp_path / "b.docx")
    touch(tmp_path / "c.txt")

    result = discover(tmp_path)

    assert result.relative_paths() == ("a.md", "c.txt")
    assert [u.filename for u in result.unsupported] == ["b.docx"]


def test_root_accepts_string_path(tmp_path) -> None:
    touch(tmp_path / "a.md")

    assert discover(str(tmp_path)).supported_count == 1


def test_result_root_is_absolute(tmp_path) -> None:
    touch(tmp_path / "a.md")

    assert discover(tmp_path).root.is_absolute()


# =============================================================================
# 4. Reporting
# =============================================================================


def test_describe_is_json_safe_and_hides_absolute_paths(tmp_path) -> None:
    touch(tmp_path / "a.md")
    touch(tmp_path / "b.docx")

    payload = describe(discover(tmp_path))

    import json

    json.dumps(payload)  # must not raise
    assert str(tmp_path) not in json.dumps(payload), "describe() leaked an absolute server path"
    assert payload["supported"] == 1
    assert payload["unsupported"] == 1
    assert payload["is_empty"] is False


def test_describe_on_empty_result(tmp_path) -> None:
    payload = describe(discover(tmp_path))

    assert payload["is_empty"] is True
    assert payload["supported"] == 0


def test_result_counts_agree_with_lists(tmp_path) -> None:
    touch(tmp_path / "a.md")
    touch(tmp_path / "b.txt")
    touch(tmp_path / "c.bin")

    result = discover(tmp_path)

    assert result.supported_count == len(result.files)
    assert result.unsupported_count == len(result.unsupported)


def test_discovered_file_is_immutable(tmp_path) -> None:
    """Frozen records: a discovery result cannot be mutated downstream."""
    touch(tmp_path / "a.md")

    item = discover(tmp_path).files[0]

    with pytest.raises(Exception):  # noqa: B017 - frozen dataclass raises FrozenInstanceError
        item.filename = "other.md"  # type: ignore[misc]


def test_records_have_expected_types(tmp_path) -> None:
    touch(tmp_path / "a.md")
    touch(tmp_path / "b.bin")

    result = discover(tmp_path)

    assert isinstance(result.files[0], DiscoveredFile)
    assert isinstance(result.unsupported[0], UnsupportedFile)
