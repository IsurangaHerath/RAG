"""Tests for the PDF loader (Stage 12).

The suite covers what the stage asks for — text extraction, page numbers,
encrypted and scanned files, and malformed input — plus the guarantees that
would otherwise rot quietly: 1-based page numbering, page-by-page failure
isolation, and the fact that a missing text layer is reported rather than
raised.

PDFs are generated at runtime by ``pdf_fixtures``; nothing is downloaded and
no binary fixture is committed. No OpenAI call is made anywhere here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.domain.errors import DocumentLoadError, NotFoundError, ValidationError
from app.ingestion import loaders as loaders_module
from app.ingestion import pdf_loader
from app.ingestion.pdf_loader import (
    PDF_MEDIA_TYPE,
    PdfDocument,
    PdfPage,
    _translate_pypdf_error,
    can_load_pdf,
    load_pdf,
)
from tests.unit.pdf_fixtures import (
    build_blank_pdf,
    build_encrypted_pdf,
    build_pdf,
    build_pdf_with_broken_page_tree,
)


def write_pdf(path: Path, data: bytes) -> Path:
    path.write_bytes(data)
    return path


@pytest.fixture
def three_page_pdf(tmp_path: Path) -> Path:
    """A valid three-page PDF with recognisable text on each page."""
    return write_pdf(tmp_path / "doc.pdf", build_pdf(["Alpha page", "Beta page", "Gamma page"]))


# =============================================================================
# 1. Successful loading
# =============================================================================


def test_loads_a_single_page_pdf(tmp_path: Path) -> None:
    path = write_pdf(tmp_path / "one.pdf", build_pdf(["Only page"]))

    doc = load_pdf(path)

    assert isinstance(doc, PdfDocument)
    assert doc.page_count == 1
    assert doc.pages[0].text == "Only page"
    assert doc.media_type == "application/pdf"
    assert doc.source_name == "one.pdf"


def test_loads_every_page_of_a_multi_page_pdf(three_page_pdf: Path) -> None:
    doc = load_pdf(three_page_pdf)

    assert doc.page_count == 3
    assert [page.text for page in doc.pages] == ["Alpha page", "Beta page", "Gamma page"]


def test_page_numbers_are_one_based(three_page_pdf: Path) -> None:
    """Citations say "page 1", not "page 0" — the loader must agree."""
    doc = load_pdf(three_page_pdf)

    assert [page.page_number for page in doc.pages] == [1, 2, 3]


def test_page_numbers_stay_positionally_aligned(three_page_pdf: Path) -> None:
    """pages[i].page_number == i + 1 is what lets Stage 18 trust the order."""
    doc = load_pdf(three_page_pdf)

    for index, page in enumerate(doc.pages):
        assert page.page_number == index + 1


def test_text_property_joins_pages(three_page_pdf: Path) -> None:
    doc = load_pdf(three_page_pdf)

    assert doc.text == "Alpha page\n\nBeta page\n\nGamma page"


def test_char_count_sums_the_pages(three_page_pdf: Path) -> None:
    doc = load_pdf(three_page_pdf)

    assert doc.char_count == sum(page.char_count for page in doc.pages)
    assert doc.char_count == len("Alpha pageBeta pageGamma page")


def test_byte_size_reflects_the_file(tmp_path: Path) -> None:
    data = build_pdf(["sized"])
    path = write_pdf(tmp_path / "sized.pdf", data)

    assert load_pdf(path).byte_size == len(data)


def test_media_type_constant() -> None:
    assert PDF_MEDIA_TYPE == "application/pdf"


def test_media_type_can_be_passed_explicitly(tmp_path: Path) -> None:
    path = write_pdf(tmp_path / "doc.pdf", build_pdf(["x"]))

    assert load_pdf(path, media_type="application/pdf").media_type == "application/pdf"


def test_summary_contains_no_document_text(tmp_path: Path) -> None:
    """Logs must never carry document content (the project's privacy rule)."""
    marker = "CONFIDENTIAL_PDF_MARKER"
    path = write_pdf(tmp_path / "doc.pdf", build_pdf([marker]))

    doc = load_pdf(path)

    assert marker not in str(doc.summary())
    assert "text" not in doc.summary()


def test_summary_is_json_serialisable(three_page_pdf: Path) -> None:
    json.dumps(load_pdf(three_page_pdf).summary())


def test_summary_reports_failed_pages_as_a_list(three_page_pdf: Path) -> None:
    """Tuples are not JSON, so the summary must convert."""
    summary = load_pdf(three_page_pdf).summary()

    assert summary["failed_pages"] == []
    assert isinstance(summary["failed_pages"], list)


def test_result_is_immutable(three_page_pdf: Path) -> None:
    doc = load_pdf(three_page_pdf)

    with pytest.raises(Exception):  # noqa: B017 - frozen dataclass raises FrozenInstanceError
        doc.pages = ()  # type: ignore[misc]


def test_page_is_immutable(three_page_pdf: Path) -> None:
    page = load_pdf(three_page_pdf).pages[0]

    with pytest.raises(Exception):  # noqa: B017
        page.text = "changed"  # type: ignore[misc]


def test_page_helpers() -> None:
    assert PdfPage(page_number=1, text="hi").char_count == 2
    assert PdfPage(page_number=1, text="  \n ").is_empty is True
    assert PdfPage(page_number=1, text="x").is_empty is False


# =============================================================================
# 2. Scanned / empty PDFs (reported, not raised)
# =============================================================================


def test_scanned_pdf_has_no_text_layer_but_is_not_an_error(tmp_path: Path) -> None:
    """A scan is a valid PDF with no text. It must be skippable, not fatal."""
    path = write_pdf(tmp_path / "scan.pdf", build_blank_pdf(2))

    doc = load_pdf(path)

    assert doc.page_count == 2
    assert doc.has_text_layer is False
    assert doc.is_empty is True
    assert doc.char_count == 0


def test_scanned_pdf_pages_are_empty_strings(tmp_path: Path) -> None:
    path = write_pdf(tmp_path / "scan.pdf", build_blank_pdf(1))

    doc = load_pdf(path)

    assert doc.pages[0].text == ""
    assert doc.pages[0].is_empty is True


def test_single_blank_page_pdf(tmp_path: Path) -> None:
    path = write_pdf(tmp_path / "blank.pdf", build_blank_pdf(1))

    assert load_pdf(path).page_count == 1


def test_has_text_layer_is_true_when_any_page_has_text(tmp_path: Path) -> None:
    """A mostly-scanned document with one typed page still has extractable text."""
    path = write_pdf(tmp_path / "mixed.pdf", build_pdf(["typed page"]))

    doc = load_pdf(path)

    assert doc.has_text_layer is True
    assert doc.is_empty is False


# =============================================================================
# 3. Text fidelity
# =============================================================================


def test_unicode_text_is_not_mangled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The loader must pass extracted text through untouched.

    PDF encoding internals are pypdf's problem; what matters here is that we
    do not break characters on the way out.
    """
    from pypdf._page import PageObject

    unicode_text = "café ☕ — naïve 日本語"
    monkeypatch.setattr(PageObject, "extract_text", lambda self, *a, **k: unicode_text)

    path = write_pdf(tmp_path / "u.pdf", build_pdf(["placeholder"]))
    doc = load_pdf(path)

    assert doc.pages[0].text == unicode_text


def test_crlf_in_extracted_text_is_normalized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pypdf._page import PageObject

    monkeypatch.setattr(
        PageObject, "extract_text", lambda self, *a, **k: "line one\r\nline two\r\n"
    )

    path = write_pdf(tmp_path / "crlf.pdf", build_pdf(["x"]))

    assert load_pdf(path).pages[0].text == "line one\nline two"


def test_lone_carriage_return_is_normalized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pypdf._page import PageObject

    monkeypatch.setattr(PageObject, "extract_text", lambda self, *a, **k: "a\rb")

    path = write_pdf(tmp_path / "cr.pdf", build_pdf(["x"]))

    assert load_pdf(path).pages[0].text == "a\nb"


def test_trailing_newline_is_stripped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """pypdf usually ends a page with a newline; it is not part of the content."""
    from pypdf._page import PageObject

    monkeypatch.setattr(PageObject, "extract_text", lambda self, *a, **k: "content\n\n")

    path = write_pdf(tmp_path / "t.pdf", build_pdf(["x"]))

    assert load_pdf(path).pages[0].text == "content"


def test_none_extraction_is_treated_as_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """pypdf returns None for some pages; that must not become the string 'None'."""
    from pypdf._page import PageObject

    monkeypatch.setattr(PageObject, "extract_text", lambda self, *a, **k: None)

    path = write_pdf(tmp_path / "n.pdf", build_pdf(["x"]))
    doc = load_pdf(path)

    assert doc.pages[0].text == ""
    assert doc.has_text_layer is False


# =============================================================================
# 4. Malformed and unreadable input
# =============================================================================


@pytest.mark.parametrize(
    ("label", "data"),
    [
        ("plain text", b"this is definitely not a pdf"),
        ("truncated", b"%PDF-1.4"),
        ("html", b"<!DOCTYPE html><html><body>hi</body></html>"),
        ("zip renamed", b"PK\x03\x04\x14\x00\x00\x00"),
        ("null bytes", b"\x00\x00\x00\x00"),
    ],
)
def test_unparseable_input_raises_invalid_pdf(tmp_path: Path, label: str, data: bytes) -> None:
    path = write_pdf(tmp_path / f"{label.replace(' ', '_')}.pdf", data)

    with pytest.raises(DocumentLoadError) as excinfo:
        load_pdf(path)

    assert excinfo.value.code == "invalid_pdf"
    assert excinfo.value.http_status == 422


def test_empty_file_raises_invalid_pdf(tmp_path: Path) -> None:
    """A zero-byte .pdf cannot be a PDF, so it is invalid rather than 'empty'."""
    path = write_pdf(tmp_path / "empty.pdf", b"")

    with pytest.raises(DocumentLoadError) as excinfo:
        load_pdf(path)

    assert excinfo.value.code == "invalid_pdf"
    assert "empty" in excinfo.value.message.lower()


def test_missing_file_raises_not_found(tmp_path: Path) -> None:
    with pytest.raises(NotFoundError) as excinfo:
        load_pdf(tmp_path / "nope.pdf")

    assert excinfo.value.code == "file_not_found"
    # The absolute path must not leak into the client-facing message.
    assert str(tmp_path) not in excinfo.value.message
    assert "nope.pdf" in excinfo.value.message


def test_directory_raises_validation_error(tmp_path: Path) -> None:
    (tmp_path / "adir").mkdir()

    with pytest.raises(ValidationError) as excinfo:
        load_pdf(tmp_path / "adir")

    assert excinfo.value.code == "path_is_a_directory"


def test_non_pdf_media_type_is_rejected(tmp_path: Path) -> None:
    path = write_pdf(tmp_path / "doc.pdf", build_pdf(["x"]))

    with pytest.raises(ValidationError) as excinfo:
        load_pdf(path, media_type="text/plain")

    assert excinfo.value.code == "unsupported_media_type"
    assert "application/pdf" in excinfo.value.message


# =============================================================================
# 5. Encrypted PDFs
# =============================================================================


def test_encrypted_without_password_raises_pdf_encrypted(tmp_path: Path) -> None:
    path = write_pdf(tmp_path / "enc.pdf", build_encrypted_pdf(["secret"], "pw123"))

    with pytest.raises(DocumentLoadError) as excinfo:
        load_pdf(path)

    assert excinfo.value.code == "pdf_encrypted"
    assert "encrypted" in excinfo.value.message.lower()


def test_encrypted_with_wrong_password_raises_distinct_code(tmp_path: Path) -> None:
    """A wrong password is its own code so a UI can re-prompt."""
    path = write_pdf(tmp_path / "enc.pdf", build_encrypted_pdf(["secret"], "pw123"))

    with pytest.raises(DocumentLoadError) as excinfo:
        load_pdf(path, password="wrong")

    assert excinfo.value.code == "pdf_wrong_password"


def test_encrypted_with_correct_password_loads(tmp_path: Path) -> None:
    path = write_pdf(tmp_path / "enc.pdf", build_encrypted_pdf(["top secret"], "pw123"))

    doc = load_pdf(path, password="pw123")

    assert doc.pages[0].text == "top secret"
    assert doc.encrypted is True


def test_unencrypted_pdf_reports_encrypted_false(three_page_pdf: Path) -> None:
    assert load_pdf(three_page_pdf).encrypted is False


# =============================================================================
# 6. Per-page failure isolation
# =============================================================================


def test_one_bad_page_does_not_discard_the_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A single corrupt page must not throw away an otherwise readable file."""
    from pypdf._page import PageObject

    calls = {"n": 0}

    def flaky(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise ValueError("corrupt page")
        return f"page {calls['n']}"

    monkeypatch.setattr(PageObject, "extract_text", flaky)

    path = write_pdf(tmp_path / "three.pdf", build_pdf(["a", "b", "c"]))
    doc = load_pdf(path)

    assert doc.failed_pages == (2,)
    assert doc.pages[1].text == ""
    # Numbering and ordering survive the failure.
    assert [page.page_number for page in doc.pages] == [1, 2, 3]
    assert doc.pages[0].text == "page 1"
    assert doc.pages[2].text == "page 3"


def test_all_pages_failing_raises_invalid_pdf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If nothing can be extracted, that is a real failure, not a scan."""
    from pypdf._page import PageObject

    def always_fail(self, *args, **kwargs):
        raise ValueError("every page is broken")

    monkeypatch.setattr(PageObject, "extract_text", always_fail)

    path = write_pdf(tmp_path / "broken.pdf", build_pdf(["a", "b"]))

    with pytest.raises(DocumentLoadError) as excinfo:
        load_pdf(path)

    assert excinfo.value.code == "invalid_pdf"


def test_failed_pages_are_reported_in_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pypdf._page import PageObject

    calls = {"n": 0}

    def flaky(self, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("boom")
        return "ok"

    monkeypatch.setattr(PageObject, "extract_text", flaky)

    path = write_pdf(tmp_path / "t.pdf", build_pdf(["a", "b"]))

    assert load_pdf(path).summary()["failed_pages"] == [1]


# =============================================================================
# 7. Loaders integration
# =============================================================================


def test_load_pdf_is_re_exported_from_loaders() -> None:
    """Stage 11's error message told users to call loaders.load_pdf, so it exists."""
    assert loaders_module.load_pdf is load_pdf
    assert "load_pdf" in loaders_module.__all__


def test_pdf_is_not_a_text_loader() -> None:
    """can_load() stays strictly about text; PDF has its own function."""
    assert loaders_module.can_load("application/pdf") is False
    assert loaders_module.can_load("text/plain") is True


def test_can_load_pdf_helper() -> None:
    assert can_load_pdf("application/pdf") is True
    assert can_load_pdf("text/plain") is False


def test_media_type_for_path_still_defers_pdf(tmp_path: Path) -> None:
    """text loaders must not silently decode PDF bytes as UTF-8."""
    with pytest.raises(ValidationError) as excinfo:
        loaders_module.media_type_for_path("doc.pdf")

    assert excinfo.value.code == "unsupported_media_type"
    assert "load_pdf" in excinfo.value.message


def test_pdf_files_are_not_loadable_by_load_text(tmp_path: Path) -> None:
    path = write_pdf(tmp_path / "doc.pdf", build_pdf(["x"]))

    with pytest.raises(ValidationError):
        loaders_module.load_text(path)


# =============================================================================
# 8. Error-mapping paths that are hard to trigger on their own
#
# These branches are real, but producing them with a genuine file is either
# impossible on Windows (POSIX mode bits do not apply) or would require a
# corrupt binary nobody can review. Forcing the underlying call exercises the
# same translation logic the real failure would hit.
# =============================================================================


def test_broken_page_tree_is_reported_as_invalid_pdf(tmp_path: Path) -> None:
    """A PDF that parses but has an unusable page tree must not crash.

    pypdf raises only when the page count is requested, so this exercises the
    page-count error path rather than the open path.
    """
    path = write_pdf(tmp_path / "broken_tree.pdf", build_pdf_with_broken_page_tree())

    with pytest.raises(DocumentLoadError) as excinfo:
        load_pdf(path)

    assert excinfo.value.code == "invalid_pdf"


def test_file_not_decrypted_maps_to_pdf_encrypted() -> None:
    """Pins the mapper's encrypted branch directly.

    Reaching this through a real file is deliberately hard: ``_decrypt`` runs
    first and raises its own clearer errors, so this branch is a safety net
    for the unusual case where pypdf refuses page access while reporting the
    file as not encrypted. Testing the mapper keeps the net from rotting.
    """
    from pypdf.errors import FileNotDecryptedError

    error = _translate_pypdf_error(FileNotDecryptedError("File has not been decrypted"), "x.pdf")

    assert error.code == "pdf_encrypted"
    assert error.http_status == 422


def test_generic_pypdf_error_becomes_invalid_pdf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Any pypdf error we do not name specifically must still be typed."""
    from pypdf.errors import PyPdfError

    class UnusualPyPdfError(PyPdfError):
        pass

    def boom(*args, **kwargs):
        raise UnusualPyPdfError("something odd")

    monkeypatch.setattr(pdf_loader, "PdfReader", boom)

    path = write_pdf(tmp_path / "odd.pdf", build_pdf(["x"]))

    with pytest.raises(DocumentLoadError) as excinfo:
        load_pdf(path)

    assert excinfo.value.code == "invalid_pdf"


def test_os_error_while_opening_pdf_is_typed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A raw OSError from opening the file must not escape."""

    def boom(*args, **kwargs):
        raise OSError("device returned an error")

    monkeypatch.setattr(pdf_loader, "PdfReader", boom)

    path = write_pdf(tmp_path / "x.pdf", build_pdf(["x"]))

    with pytest.raises(DocumentLoadError) as excinfo:
        load_pdf(path)

    assert excinfo.value.code == "unreadable_file"


def test_stat_failure_is_typed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failing stat (e.g. a file removed between checks) must be typed too."""
    path = write_pdf(tmp_path / "racy.pdf", build_pdf(["x"]))
    real_stat = Path.stat

    def flaky_stat(self: Path, *args, **kwargs):
        if self.name == "racy.pdf":
            raise OSError("stat failed")
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", flaky_stat)

    with pytest.raises(DocumentLoadError) as excinfo:
        load_pdf(path)

    assert excinfo.value.code == "unreadable_file"
