"""Tests for TXT and Markdown loaders (Stage 11).

Covers the four cases the stage requires — **successful load, empty file,
Unicode content, malformed input** — plus line-ending normalisation and the
typed-error contract for unreadable files.

Nothing here touches the network or a real API key.
"""

from __future__ import annotations

import codecs
from pathlib import Path

import pytest

from app.domain.errors import DocumentLoadError, NotFoundError, ValidationError
from app.ingestion.loaders import (
    TEXT_MEDIA_TYPES,
    TextDocument,
    can_load,
    detect_newline_style,
    load,
    load_text,
    media_type_for_path,
)


def write_bytes(path: Path, data: bytes) -> Path:
    path.write_bytes(data)
    return path


def write_text(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


# =============================================================================
# 1. Successful loading
# =============================================================================


def test_load_txt_file(tmp_path: Path) -> None:
    path = write_text(tmp_path / "notes.txt", "line one\nline two\n")

    doc = load_text(path)

    assert doc.text == "line one\nline two\n"
    assert doc.media_type == "text/plain"
    assert doc.source_name == "notes.txt"
    assert doc.char_count == len("line one\nline two\n")
    assert doc.line_count == 3
    assert doc.is_empty is False


def test_load_markdown_preserves_structure(tmp_path: Path) -> None:
    """Headings, fences and indentation must survive byte-for-byte.

    Chunking (Stage 18) relies on this structure, so a loader that tidied
    text would quietly damage retrieval later.
    """
    markdown = (
        "# Title\n"
        "\n"
        "Intro paragraph.\n"
        "\n"
        "```python\n"
        "def f():\n"
        "    return 1\n"
        "```\n"
        "\n"
        "- item one\n"
        "- item two\n"
    )
    path = write_text(tmp_path / "doc.md", markdown)

    doc = load_text(path)

    assert doc.text == markdown
    assert doc.media_type == "text/markdown"


def test_leading_indentation_preserved(tmp_path: Path) -> None:
    text = "    indented code block\n    second line\n"
    path = write_text(tmp_path / "a.txt", text)

    assert load_text(path).text == text


def test_blank_lines_preserved(tmp_path: Path) -> None:
    text = "para one\n\n\npara two\n"
    path = write_text(tmp_path / "a.txt", text)

    assert load_text(path).text == text


def test_media_type_inferred_from_markdown_suffix(tmp_path: Path) -> None:
    path = write_text(tmp_path / "README.md", "hi")

    assert load_text(path).media_type == "text/markdown"


def test_media_type_inferred_from_txt_suffix(tmp_path: Path) -> None:
    path = write_text(tmp_path / "notes.txt", "hi")

    assert load_text(path).media_type == "text/plain"


def test_media_type_can_be_overridden(tmp_path: Path) -> None:
    path = write_text(tmp_path / "noext", "hi")

    assert load_text(path, media_type="text/markdown").media_type == "text/markdown"


def test_load_alias_behaves_like_load_text(tmp_path: Path) -> None:
    path = write_text(tmp_path / "a.txt", "hello")

    assert load(path).text == load_text(path).text


def test_absolute_and_relative_paths_both_work(tmp_path: Path, monkeypatch) -> None:
    write_text(tmp_path / "a.txt", "hello")
    monkeypatch.chdir(tmp_path)

    assert load_text(Path("a.txt")).text == "hello"
    assert load_text(tmp_path / "a.txt").text == "hello"


def test_summary_contains_no_document_text(tmp_path: Path) -> None:
    """Logs must never carry document content (privacy rule)."""
    secret = "CONFIDENTIAL INTERNAL CONTENT"
    path = write_text(tmp_path / "a.txt", secret)

    doc = load_text(path)

    assert secret not in str(doc.summary())
    assert "text" not in doc.summary()


def test_summary_is_json_serialisable(tmp_path: Path) -> None:
    import json

    path = write_text(tmp_path / "a.txt", "hello")
    json.dumps(load_text(path).summary())


# =============================================================================
# 2. Empty files
# =============================================================================


def test_zero_byte_file_is_empty_not_an_error(tmp_path: Path) -> None:
    """Empty is valid — Stage 14 marks it 'skipped', not 'failed'."""
    path = write_bytes(tmp_path / "empty.txt", b"")

    doc = load_text(path)

    assert doc.text == ""
    assert doc.is_empty is True
    assert doc.line_count == 0
    assert doc.char_count == 0
    assert doc.byte_size == 0


def test_whitespace_only_file_is_empty(tmp_path: Path) -> None:
    path = write_bytes(tmp_path / "blank.txt", b"   \n\t\n  \n")

    doc = load_text(path)

    assert doc.text == "   \n\t\n  \n"  # content preserved...
    assert doc.is_empty is True  # ...but it counts as empty


def test_file_with_only_newlines_is_empty(tmp_path: Path) -> None:
    path = write_bytes(tmp_path / "newlines.txt", b"\n\n\n")

    doc = load_text(path)

    assert doc.is_empty is True
    assert doc.line_count == 4


def test_single_line_no_trailing_newline(tmp_path: Path) -> None:
    path = write_bytes(tmp_path / "one.txt", b"only line")

    doc = load_text(path)

    assert doc.line_count == 1
    assert doc.is_empty is False


# =============================================================================
# 3. Unicode content
# =============================================================================


def test_unicode_text_round_trips_exactly(tmp_path: Path) -> None:
    text = "Héllo wörld — café ☕\n日本語テキスト\nΕλληνικά\n"
    path = write_text(tmp_path / "unicode.txt", text)

    doc = load_text(path)

    assert doc.text == text
    assert doc.encoding == "utf-8"
    assert doc.had_bom is False


def test_emoji_and_astral_plane_characters(tmp_path: Path) -> None:
    """4-byte code points exercise the surrogatepair path in codecs."""
    text = "rocket 🚀 and 👩‍💻 family\n"
    path = write_text(tmp_path / "emoji.txt", text)

    assert load_text(path).text == text


def test_utf8_bom_is_stripped_but_recorded(tmp_path: Path) -> None:
    """A BOM would otherwise show up as U+FEFF at the start of the first chunk.

    The bytes are built with ``codecs.BOM_UTF8`` rather than
    ``"...".encode("utf-8-sig")``: the latter would add a BOM *and* keep any
    U+FEFF already in the string, producing a double-BOM fixture that does
    not resemble a real file.
    """
    raw = codecs.BOM_UTF8 + b"BOM file content"
    path = write_bytes(tmp_path / "bom.txt", raw)

    doc = load_text(path)

    assert doc.had_bom is True
    assert doc.encoding == "utf-8-sig"
    assert doc.text == "BOM file content"
    assert not doc.text.startswith("﻿")


def test_no_bom_reports_plain_utf8(tmp_path: Path) -> None:
    path = write_text(tmp_path / "plain.txt", "content")

    doc = load_text(path)

    assert doc.had_bom is False
    assert doc.encoding == "utf-8"


def test_unicode_filename_loads(tmp_path: Path) -> None:
    path = write_text(tmp_path / "café-日本語.md", "# Ünïcödé\n")

    doc = load_text(path)

    assert doc.source_name == "café-日本語.md"
    assert doc.text == "# Ünïcödé\n"


# =============================================================================
# 4. Malformed / unreadable input
# =============================================================================


def test_invalid_utf8_raises_typed_error_with_offset(tmp_path: Path) -> None:
    """Strict decoding: corrupt bytes must not be silently indexed."""
    path = write_bytes(tmp_path / "bad.txt", b"good text \xff\xfe bad text")

    with pytest.raises(DocumentLoadError) as excinfo:
        load_text(path)

    assert excinfo.value.code == "encoding_error"
    assert excinfo.value.http_status == 422
    # The actionable hint is in the user-facing message.
    assert "UTF-8" in excinfo.value.message
    # The offset goes in developer detail, not the client message.
    assert "offset" in (excinfo.value.detail or "")


def test_binary_content_masquerading_as_txt_is_rejected(tmp_path: Path) -> None:
    """A .docx renamed to .txt contains NULs and must not be indexed as text."""
    path = write_bytes(tmp_path / "fake.txt", b"PK\x03\x04\x00\x00binary\x00junk")

    with pytest.raises(DocumentLoadError) as excinfo:
        load_text(path)

    assert excinfo.value.code == "binary_content"
    assert "binary" in excinfo.value.message


def test_empty_bytes_do_not_trigger_binary_detection(tmp_path: Path) -> None:
    """Guard against NUL-check misfiring on empty content."""
    path = write_bytes(tmp_path / "empty.txt", b"")

    assert load_text(path).text == ""


def test_missing_file_raises_not_found(tmp_path: Path) -> None:
    with pytest.raises(NotFoundError) as excinfo:
        load_text(tmp_path / "nope.txt")

    assert excinfo.value.code == "file_not_found"
    # Absolute path must not leak into the client-facing message.
    assert str(tmp_path) not in excinfo.value.message
    assert "nope.txt" in excinfo.value.message


def test_directory_instead_of_file_raises_validation_error(tmp_path: Path) -> None:
    (tmp_path / "adir").mkdir()

    with pytest.raises(ValidationError) as excinfo:
        load_text(tmp_path / "adir")

    assert excinfo.value.code == "path_is_a_directory"


def test_permission_denied_raises_typed_error(tmp_path: Path) -> None:
    path = write_text(tmp_path / "locked.txt", "secret")

    import os

    os.chmod(path, 0o000)
    try:
        # Windows denies read via sharing flags rather than mode bits, so the
        # mode change may not actually block us. Only assert when it does.
        try:
            path.read_bytes()
        except PermissionError:
            with pytest.raises(DocumentLoadError) as excinfo:
                load_text(path)
            assert excinfo.value.code == "permission_denied"
    finally:
        os.chmod(path, 0o644)


def test_os_error_is_mapped_to_a_typed_error(tmp_path: Path, monkeypatch) -> None:
    """Force the OS failure path directly.

    Windows does not honour POSIX mode bits, so a real permission failure is
    not reproducible here. Patching ``Path.read_bytes`` exercises the same
    translation logic that a genuine ``PermissionError`` would hit.
    """
    from pathlib import Path as PathCls

    path = tmp_path / "x.txt"
    path.write_text("hi", encoding="utf-8")

    def deny(self, *args, **kwargs):
        raise PermissionError(13, "Access is denied")

    monkeypatch.setattr(PathCls, "read_bytes", deny)

    with pytest.raises(DocumentLoadError) as excinfo:
        load_text(path)

    assert excinfo.value.code == "permission_denied"
    assert excinfo.value.http_status == 422


def test_generic_os_error_is_mapped_to_a_typed_error(tmp_path: Path, monkeypatch) -> None:
    """A non-permission I/O failure must not escape as a raw OSError."""
    from pathlib import Path as PathCls

    path = tmp_path / "x.txt"
    path.write_text("hi", encoding="utf-8")

    def fail(self, *args, **kwargs):
        raise OSError("device returned an error")

    monkeypatch.setattr(PathCls, "read_bytes", fail)

    with pytest.raises(DocumentLoadError) as excinfo:
        load_text(path)

    assert excinfo.value.code == "unreadable_file"
    assert "could not read x.txt" in excinfo.value.message


def test_unsupported_media_type_rejected(tmp_path: Path) -> None:
    path = write_text(tmp_path / "a.txt", "hi")

    with pytest.raises(ValidationError) as excinfo:
        load_text(path, media_type="application/pdf")

    assert excinfo.value.code == "unsupported_media_type"
    assert "load_text" in excinfo.value.message


def test_pdf_path_is_deferred_to_the_pdf_loader(tmp_path: Path) -> None:
    """media_type_for_path fails loudly rather than decoding PDF bytes as text."""
    path = tmp_path / "doc.pdf"
    path.write_bytes(b"%PDF-1.7 not really text")

    with pytest.raises(ValidationError) as excinfo:
        load_text(path)

    assert excinfo.value.code == "unsupported_media_type"
    assert "load_pdf" in excinfo.value.message


# =============================================================================
# 5. Line ending normalisation
# =============================================================================


def test_crlf_is_normalized_to_lf(tmp_path: Path) -> None:
    path = write_bytes(tmp_path / "win.txt", b"alpha\r\nbeta\r\n")

    doc = load_text(path)

    assert doc.text == "alpha\nbeta\n"
    assert doc.original_newlines == "crlf"
    assert doc.newlines_normalized is True
    assert doc.line_count == 3


def test_crlf_normalization_does_not_double_newlines(tmp_path: Path) -> None:
    """Naive replace('\\r', '\\n') would turn CRLF into blank lines."""
    path = write_bytes(tmp_path / "win.txt", b"a\r\nb\r\nc")

    assert load_text(path).text == "a\nb\nc"


def test_old_mac_cr_is_normalized(tmp_path: Path) -> None:
    path = write_bytes(tmp_path / "old.txt", b"alpha\rbeta\r")

    doc = load_text(path)

    assert doc.text == "alpha\nbeta\n"
    assert doc.original_newlines == "cr"
    assert doc.newlines_normalized is True


def test_lf_is_left_alone(tmp_path: Path) -> None:
    path = write_bytes(tmp_path / "unix.txt", b"alpha\nbeta\n")

    doc = load_text(path)

    assert doc.text == "alpha\nbeta\n"
    assert doc.original_newlines == "lf"
    assert doc.newlines_normalized is False


def test_mixed_line_endings_are_all_normalized(tmp_path: Path) -> None:
    path = write_bytes(tmp_path / "mixed.txt", b"a\r\nb\nc\rd")

    doc = load_text(path)

    assert doc.text == "a\nb\nc\nd"
    assert doc.original_newlines == "mixed"
    assert doc.newlines_normalized is True


def test_normalization_can_be_disabled(tmp_path: Path) -> None:
    """Diagnostics need to inspect the file exactly as stored."""
    path = write_bytes(tmp_path / "win.txt", b"a\r\nb")

    doc = load_text(path, normalize_newlines=False)

    assert doc.text == "a\r\nb"
    assert doc.original_newlines == "crlf"
    assert doc.newlines_normalized is False


def test_file_without_newlines_reports_none(tmp_path: Path) -> None:
    assert detect_newline_style(b"one line only") == "none"


def test_detect_newline_style_variants() -> None:
    assert detect_newline_style(b"") == "none"
    assert detect_newline_style(b"a\nb") == "lf"
    assert detect_newline_style(b"a\r\nb") == "crlf"
    assert detect_newline_style(b"a\rb") == "cr"
    assert detect_newline_style(b"a\r\nb\nc") == "mixed"


# =============================================================================
# 6. Helpers
# =============================================================================


def test_media_type_for_path_defaults_to_plain() -> None:
    assert media_type_for_path("whatever.xyz") == "text/plain"
    assert media_type_for_path("notes.txt") == "text/plain"
    assert media_type_for_path("notes.MD") == "text/markdown"


def test_can_load_matches_supported_types() -> None:
    assert can_load("text/plain") is True
    assert can_load("text/markdown") is True
    assert can_load("application/pdf") is False


def test_text_media_types_constant() -> None:
    assert TEXT_MEDIA_TYPES == frozenset({"text/plain", "text/markdown"})


def test_result_is_immutable(tmp_path: Path) -> None:
    path = write_text(tmp_path / "a.txt", "x")

    with pytest.raises(Exception):  # noqa: B017 - frozen dataclass raises FrozenInstanceError
        load_text(path).text = "other"  # type: ignore[misc]


def test_text_document_type(tmp_path: Path) -> None:
    path = write_text(tmp_path / "a.txt", "x")

    assert isinstance(load_text(path), TextDocument)
