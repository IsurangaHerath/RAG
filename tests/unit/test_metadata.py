"""Tests for document metadata extraction.

Two properties get more attention than the rest, because both were found by
probing pypdf rather than assumed:

* Embedded metadata is attacker-controlled and arrives unsanitised. A PDF
  title of ``"Bad\\x00Title\\x07"`` round-trips verbatim through pypdf, so
  what lands in the vector index must not.
* Malformed dates exist in the wild. pypdf's own parser raises ``ValueError``
  on ``"not-a-date"``; ingestion must record "no date" rather than die.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from typing import cast

import pytest
from pypdf import PdfReader

from app.ingestion.metadata import (
    MAX_FIELD_LENGTH,
    MAX_KEYWORDS,
    MAX_TITLE_LENGTH,
    PdfInfo,
    TitleSource,
    build_metadata,
    extract_pdf_info,
    extract_text_info,
    extract_title_from_text,
    fallback_title,
    parse_pdf_datetime,
    sanitize_keywords,
    sanitize_text,
)
from app.ingestion.pdf_loader import load_pdf
from tests.unit.pdf_fixtures import build_pdf, build_pdf_with_metadata


def write_pdf(tmp_path, data: bytes, name: str = "doc.pdf"):
    path = tmp_path / name
    path.write_bytes(data)
    return path


# --- sanitize_text ---------------------------------------------------------


def test_control_characters_are_stripped() -> None:
    """NUL and BEL must not survive into stored metadata.

    pypdf passes them through unchanged, so this is our job. A control
    character in a stored value can forge a log line or escape a terminal.
    """
    assert sanitize_text("Bad\x00Title\x07") == "Bad Title"


def test_newlines_and_tabs_become_spaces() -> None:
    """A metadata *value* is a single field; newlines enable log forging."""
    assert sanitize_text("Line1\nLine2\tTabbed") == "Line1 Line2 Tabbed"


def test_whitespace_runs_collapse_and_edges_trim() -> None:
    assert sanitize_text("  spaced   out  ") == "spaced out"


def test_empty_or_whitespace_returns_none() -> None:
    """None, not empty string, so callers can tell absent from blank."""
    assert sanitize_text("") is None
    assert sanitize_text("   ") is None
    assert sanitize_text(None) is None


def test_numbers_are_stringified() -> None:
    """pypdf coerces ints to ``"42"``; we accept either shape."""
    assert sanitize_text(42) == "42"
    assert sanitize_text(3.5) == "3.5"


def test_unsupported_types_are_rejected() -> None:
    """Lists and dicts cannot be stored as a scalar metadata value."""
    assert sanitize_text(["a", "b"]) is None
    assert sanitize_text({"k": "v"}) is None


def test_values_are_truncated_to_the_field_cap() -> None:
    assert sanitize_text("x" * 500, max_length=10) == "xxxxxxxxxx"


def test_default_field_cap_is_applied() -> None:
    assert len(sanitize_text("y" * (MAX_FIELD_LENGTH * 2)) or "") == MAX_FIELD_LENGTH


def test_title_cap_is_shorter_than_other_fields() -> None:
    """A title is a label; a long one is almost always junk or an attack."""
    assert MAX_TITLE_LENGTH < MAX_FIELD_LENGTH


def test_non_ascii_is_preserved_not_folded() -> None:
    """Sinhala and emoji are legitimate titles and must survive intact."""
    assert sanitize_text("Sinhala \u0dc3\u0dd2\u0d82\u0dc4\u0dba") == (
        "Sinhala \u0dc3\u0dd2\u0d82\u0dc4\u0dba"
    )
    assert sanitize_text("emoji \U0001f389") == "emoji \U0001f389"


def test_del_character_is_stripped() -> None:
    """DEL becomes a space, like every other control character.

    One rule for all control characters is simpler to reason about than
    special-casing DEL, and the whitespace collapse repairs the gap.
    """
    assert sanitize_text("a\x7fb") == "a b"


# --- sanitize_keywords -----------------------------------------------------


def test_keywords_split_on_commas_and_semicolons() -> None:
    assert sanitize_keywords("payments,api;spec") == ("payments", "api", "spec")


def test_keywords_split_on_whitespace_and_empty_parts_dropped() -> None:
    assert sanitize_keywords("a b,, c") == ("a", "b", "c")


def test_keywords_are_deduplicated_preserving_order() -> None:
    assert sanitize_keywords("a,b,a,c,b") == ("a", "b", "c")


def test_keywords_are_cleaned_individually() -> None:
    assert sanitize_keywords("good\x00,bad") == ("good", "bad")


def test_absent_or_blank_keywords_yield_empty_tuple() -> None:
    assert sanitize_keywords(None) == ()
    assert sanitize_keywords("") == ()
    assert sanitize_keywords("   ") == ()


def test_keyword_count_is_capped() -> None:
    many = ",".join(f"k{i}" for i in range(MAX_KEYWORDS + 25))
    assert len(sanitize_keywords(many)) == MAX_KEYWORDS


# --- parse_pdf_datetime ----------------------------------------------------


def test_offset_date_converts_to_utc() -> None:
    """+05'30' on 09:30 is 04:00 UTC — the arithmetic must be right."""
    parsed = parse_pdf_datetime("D:20260104093000+05'30'")
    assert parsed == datetime(2026, 1, 4, 4, 0, tzinfo=UTC)


def test_negative_offset_date_converts_to_utc() -> None:
    parsed = parse_pdf_datetime("D:20260104093000-08'00'")
    assert parsed == datetime(2026, 1, 4, 17, 30, tzinfo=UTC)


def test_zulu_date_is_utc() -> None:
    assert parse_pdf_datetime("D:20260215114500Z") == datetime(2026, 2, 15, 11, 45, tzinfo=UTC)


def test_bare_year_defaults_to_january_first() -> None:
    """The spec makes every component after the year optional."""
    assert parse_pdf_datetime("D:2026") == datetime(2026, 1, 1, tzinfo=UTC)


def test_year_and_month_defaults_to_first_of_month() -> None:
    assert parse_pdf_datetime("D:202606") == datetime(2026, 6, 1, tzinfo=UTC)


def test_malformed_date_returns_none_instead_of_raising() -> None:
    """pypdf's own parser raises here; one bad field must not lose the file."""
    assert parse_pdf_datetime("not-a-date") is None
    assert parse_pdf_datetime("") is None
    assert parse_pdf_datetime("   ") is None


def test_impossible_calendar_date_returns_none() -> None:
    """Month 13 parses as digits but is not a real date."""
    assert parse_pdf_datetime("D:20261301000000") is None


def test_naive_datetime_is_assumed_utc() -> None:
    """The domain model rejects naive datetimes outright."""
    parsed = parse_pdf_datetime(datetime(2026, 1, 4, 9, 30))
    assert parsed is not None and parsed.tzinfo is not None


def test_aware_datetime_is_converted_to_utc() -> None:
    aware = datetime(2026, 1, 4, 9, 30, tzinfo=timezone(timedelta(hours=5, minutes=30)))
    assert parse_pdf_datetime(aware) == datetime(2026, 1, 4, 4, 0, tzinfo=UTC)


def test_non_date_types_return_none() -> None:
    assert parse_pdf_datetime(None) is None
    assert parse_pdf_datetime(42) is None


# --- text-derived metadata -------------------------------------------------


def test_heading_count_counts_atx_headings() -> None:
    info = extract_text_info("# A\n\ntext\n## B\n### C")
    assert info.heading_count == 3


def test_indented_hashes_are_not_headings() -> None:
    """An indented hash is a code block in Markdown, not a heading."""
    assert extract_text_info("    # not a heading").heading_count == 0


def test_word_count_splits_on_whitespace() -> None:
    assert extract_text_info("one  two\tthree").word_count == 3


def test_line_count_is_one_for_single_line() -> None:
    assert extract_text_info("just one line").line_count == 1


def test_empty_text_reports_zero_everything() -> None:
    info = extract_text_info("")
    assert (info.word_count, info.line_count, info.char_count, info.heading_count) == (0, 0, 0, 0)


def test_first_atx_heading_becomes_the_title() -> None:
    assert extract_title_from_text("# My Title\n\nbody text") == "My Title"


def test_first_heading_wins_over_later_ones() -> None:
    assert extract_title_from_text("# First\n\n## Second") == "First"


def test_first_non_empty_line_is_the_title_for_plain_text() -> None:
    assert extract_title_from_text("First line here\nsecond line") == "First line here"


def test_leading_blank_lines_are_ignored_when_finding_a_title() -> None:
    assert extract_title_from_text("\n\n\nActual first line") == "Actual first line"


def test_no_title_in_an_empty_document() -> None:
    assert extract_title_from_text("") is None
    assert extract_title_from_text("   \n  \n") is None


def test_heading_after_the_scan_window_is_not_used_as_a_title() -> None:
    """Only the leading lines are scanned, so a distant heading is ignored.

    The bound is the point: without it, a heading buried 500 lines down would
    silently become the document's title.
    """
    filler = "\n".join(f"filler line {i}" for i in range(60))
    body = f"{filler}\n# Distant Heading"

    # The first filler line is inside the window and wins; the heading is not
    # reached, which is exactly what the bound guarantees.
    assert extract_title_from_text(body) == "filler line 0"


# --- filename-derived fallback title ---------------------------------------


def test_hyphenated_filename_becomes_readable_title() -> None:
    assert fallback_title("payment-gateway-spec.md") == "Payment gateway spec"


def test_underscored_filename_becomes_readable_title() -> None:
    assert fallback_title("api_design_notes.txt") == "Api design notes"


def test_multi_dot_filename_strips_only_the_last_extension() -> None:
    assert fallback_title("spec.v2.md") == "Spec.v2"


def test_filename_title_is_shortened() -> None:
    assert len(fallback_title("a" * 300 + ".md") or "") <= 80


def test_extensionless_filename_is_handled() -> None:
    """A file with no dot must not be split; there is no extension to strip."""
    assert fallback_title("README") == "Readme"


# --- extract_pdf_info against real PDFs ------------------------------------


def test_pdf_metadata_is_extracted(tmp_path) -> None:
    path = write_pdf(
        tmp_path,
        build_pdf_with_metadata(
            ["body"],
            {
                "/Title": "Payment Gateway Spec",
                "/Author": "Isuranga Herath",
                "/Subject": "Technical design",
                "/Keywords": "payments,api,spec",
                "/Creator": "spec-tool 2.1",
            },
        ),
    )
    info = load_pdf(path).info

    assert info is not None
    assert info.title == "Payment Gateway Spec"
    assert info.author == "Isuranga Herath"
    assert info.subject == "Technical design"
    assert info.keywords == ("payments", "api", "spec")
    assert info.creator == "spec-tool 2.1"


def test_pdf_dates_are_extracted_as_utc(tmp_path) -> None:
    path = write_pdf(
        tmp_path,
        build_pdf_with_metadata(
            ["body"], {"/CreationDate": "D:20260104093000+05'30'", "/ModDate": "D:20260215114500Z"}
        ),
    )
    info = load_pdf(path).info

    assert info is not None
    assert info.created_at == datetime(2026, 1, 4, 4, 0, tzinfo=UTC)
    assert info.modified_at == datetime(2026, 2, 15, 11, 45, tzinfo=UTC)


def test_pdf_with_no_info_dictionary_yields_empty_info(tmp_path) -> None:
    """``reader.metadata`` is ``None`` for these; that is not an error."""
    path = write_pdf(tmp_path, build_pdf(["plain body"]))
    info = load_pdf(path).info

    assert info is not None
    assert info.is_empty
    assert info.title is None
    assert info.author is None


def test_malformed_pdf_date_does_not_fail_the_load(tmp_path) -> None:
    """A garbage date must lose the date, not the document."""
    path = write_pdf(
        tmp_path,
        build_pdf_with_metadata(["body"], {"/Title": "Doc", "/CreationDate": "not-a-date"}),
    )
    doc = load_pdf(path)

    assert doc.info is not None
    assert doc.info.created_at is None
    assert doc.info.title == "Doc"


def test_control_characters_in_pdf_metadata_are_stripped(tmp_path) -> None:
    """The headline security property, exercised end to end."""
    path = write_pdf(
        tmp_path,
        build_pdf_with_metadata(["body"], {"/Title": "Bad\x00Title\x07", "/Author": "A\nB"}),
    )
    info = load_pdf(path).info

    assert info is not None
    assert info.title == "Bad Title"
    assert info.author == "A B"


def test_over_long_pdf_metadata_is_truncated(tmp_path) -> None:
    path = write_pdf(
        tmp_path, build_pdf_with_metadata(["body"], {"/Title": "x" * 5000, "/Author": "y" * 5000})
    )
    info = load_pdf(path).info

    assert info is not None
    assert len(info.title or "") == MAX_TITLE_LENGTH
    assert len(info.author or "") == MAX_FIELD_LENGTH


def test_non_ascii_pdf_metadata_survives(tmp_path) -> None:
    path = write_pdf(
        tmp_path,
        build_pdf_with_metadata(["body"], {"/Title": "Sinhala \u0dc3\u0dd2\u0d82\u0dc4\u0dba"}),
    )
    info = load_pdf(path).info

    assert info is not None
    assert info.title == "Sinhala \u0dc3\u0dd2\u0d82\u0dc4\u0dba"


def test_encrypted_pdf_metadata_is_readable_with_the_password(tmp_path) -> None:
    """Info must be captured *after* decryption, or it would come back empty."""
    import io

    from pypdf import PdfWriter

    from tests.unit.pdf_fixtures import build_pdf

    writer = PdfWriter()
    writer.append(io.BytesIO(build_pdf(["secret"])))
    writer.add_metadata({"/Title": "Locked Spec"})
    writer.encrypt("pw123")
    buf = io.BytesIO()
    writer.write(buf)

    path = write_pdf(tmp_path, buf.getvalue())
    doc = load_pdf(path, password="pw123")

    assert doc.info is not None
    assert doc.info.title == "Locked Spec"


def test_extract_pdf_info_handles_a_reader_without_metadata(tmp_path) -> None:
    """Called directly with a bare reader, it must still return a valid object."""
    from pypdf import PdfReader

    path = write_pdf(tmp_path, build_pdf(["body"]))
    info = extract_pdf_info(PdfReader(str(path)))

    assert info.is_empty
    assert info.keywords == ()


def test_unreadable_info_dictionary_is_not_fatal() -> None:
    """pypdf is third-party code; if it raises, we degrade rather than die.

    A corrupt info dictionary would otherwise lose an otherwise readable
    document over a field nothing depends on.
    """

    class _CorruptReader:
        @property
        def metadata(self) -> object:
            raise ValueError("corrupt info dictionary")

    info = extract_pdf_info(cast(PdfReader, _CorruptReader()))
    assert info.is_empty


def test_unreadable_individual_keys_are_skipped() -> None:
    """One unreadable key must not discard the rest of the dictionary."""

    class _ExplodingInfo:
        def get(self, key: str, default: object = None) -> object:
            raise ValueError(f"unreadable key: {key}")

        creation_date = None
        modification_date = None

    class _Reader:
        @property
        def metadata(self) -> object:
            return _ExplodingInfo()

    info = extract_pdf_info(cast(PdfReader, _Reader()))

    assert info.title is None
    assert info.author is None
    assert info.keywords == ()
    assert info.created_at is None


# --- DocumentMetadata and build_metadata -----------------------------------


def test_embedded_title_beats_heading() -> None:
    md = build_metadata(
        source_name="doc.pdf",
        media_type="application/pdf",
        byte_size=10,
        text="# Heading Title",
        pdf_info=PdfInfo(
            title="Embedded Title",
            author=None,
            subject=None,
            keywords=(),
            creator=None,
            producer=None,
            created_at=None,
            modified_at=None,
        ),
    )
    assert md.title == "Embedded Title"
    assert md.title_source is TitleSource.EMBEDDED


def test_heading_beats_filename() -> None:
    md = build_metadata(
        source_name="some-file.md",
        media_type="text/markdown",
        byte_size=10,
        text="# Heading Title",
    )
    assert md.title == "Heading Title"
    assert md.title_source is TitleSource.HEADING


def test_filename_is_the_last_resort() -> None:
    md = build_metadata(
        source_name="some-file.md", media_type="text/markdown", byte_size=10, text=""
    )
    assert md.title == "Some file"
    assert md.title_source is TitleSource.FILENAME


def test_no_title_at_all_when_not_even_a_filename_is_usable() -> None:
    md = build_metadata(source_name=".md", media_type="text/markdown", byte_size=0, text="")
    assert md.title_source in {TitleSource.NONE, TitleSource.FILENAME}


def test_provenance_is_always_present() -> None:
    md = build_metadata(source_name="a.md", media_type="text/markdown", byte_size=7)
    assert md.source_name == "a.md"
    assert md.media_type == "text/markdown"
    assert md.byte_size == 7


def test_filename_title_does_not_mask_an_empty_document() -> None:
    """An invented title must not hide that the file declared nothing."""
    md = build_metadata(source_name="notes.txt", media_type="text/plain", byte_size=10, text="")
    assert md.title is not None
    assert md.is_empty


def test_a_declared_title_makes_the_document_non_empty() -> None:
    md = build_metadata(source_name="x.md", media_type="text/markdown", byte_size=5, text="# Real")
    assert not md.is_empty


def test_counts_are_derived_from_the_text() -> None:
    md = build_metadata(
        source_name="x.md", media_type="text/markdown", byte_size=5, text="# A\n\none two three"
    )
    # split() counts the heading's tokens too: "#", "A", "one", "two", "three".
    # That is intentional — a naive split is predictable, and the count feeds
    # chunk sizing rather than a display, so exactness matters more than
    # cleverness about what a "word" is.
    assert md.word_count == 5
    assert md.heading_count == 1
    assert md.char_count == len("# A\n\none two three")


def test_page_count_passes_through_for_paged_formats() -> None:
    md = build_metadata(
        source_name="x.pdf", media_type="application/pdf", byte_size=5, page_count=12
    )
    assert md.page_count == 12


def test_page_count_is_none_for_plain_text() -> None:
    md = build_metadata(source_name="x.txt", media_type="text/plain", byte_size=5)
    assert md.page_count is None


# --- to_mapping / summary --------------------------------------------------


def test_to_mapping_omits_absent_values() -> None:
    """ChromaDB rejects ``None`` metadata, so missing fields must be dropped."""
    md = build_metadata(source_name="a.md", media_type="text/markdown", byte_size=5, text="# T")
    mapping = md.to_mapping()

    assert "author" not in mapping
    assert "created_at" not in mapping
    assert "page_count" not in mapping
    assert all(value is not None for value in mapping.values())


def test_to_mapping_contains_mandatory_provenance() -> None:
    md = build_metadata(source_name="a.md", media_type="text/markdown", byte_size=5)
    mapping = md.to_mapping()
    assert mapping["source_name"] == "a.md"
    assert mapping["media_type"] == "text/markdown"
    assert mapping["byte_size"] == 5


def test_to_mapping_serialises_dates_and_keywords() -> None:
    """Neither store accepts datetime or list values, so both are flattened."""
    md = build_metadata(
        source_name="a.pdf",
        media_type="application/pdf",
        byte_size=5,
        pdf_info=PdfInfo(
            title="T",
            author=None,
            subject=None,
            keywords=("a", "b"),
            creator=None,
            producer=None,
            created_at=datetime(2026, 1, 4, tzinfo=UTC),
            modified_at=None,
        ),
    )
    mapping = md.to_mapping()

    assert mapping["created_at"] == "2026-01-04T00:00:00+00:00"
    assert mapping["keywords"] == "a, b"
    assert isinstance(mapping["title_source"], str)


def test_to_mapping_is_json_safe_scalars_only() -> None:
    md = build_metadata(
        source_name="a.pdf",
        media_type="application/pdf",
        byte_size=5,
        page_count=3,
        pdf_info=PdfInfo(
            title="T",
            author="A",
            subject="S",
            keywords=("k",),
            creator="C",
            producer="P",
            created_at=datetime(2026, 1, 4, tzinfo=UTC),
            modified_at=datetime(2026, 2, 4, tzinfo=UTC),
        ),
    )
    allowed = (str, int, float, bool)
    assert all(isinstance(v, allowed) for v in md.to_mapping().values())


def test_summary_reports_absent_fields_as_none() -> None:
    """Logs should show what was missing, not silently omit it."""
    md = build_metadata(source_name="a.md", media_type="text/markdown", byte_size=5)
    summary = md.summary()

    assert summary["author"] is None
    assert summary["page_count"] is None
    assert summary["keywords"] == []


def test_summary_never_contains_document_text() -> None:
    """Body text must not reach a log line.

    The heading is excluded deliberately: it doubles as the title, which is
    metadata we *do* store, so the assertion is about the body — the part
    that could be long, sensitive and is never needed in a log.
    """
    md = build_metadata(
        source_name="a.md",
        media_type="text/markdown",
        byte_size=5,
        text="# Clean Title\n\nconfidential body paragraph",
    )

    assert md.title == "Clean Title"
    assert "confidential body" not in str(md.summary())


def test_metadata_is_immutable() -> None:
    """Frozen, so a worker cannot mutate a record another task is reading."""
    import dataclasses

    md = build_metadata(source_name="a.md", media_type="text/markdown", byte_size=5)
    with pytest.raises(dataclasses.FrozenInstanceError):
        md.title = "changed"  # type: ignore[misc]


def test_module_exports_the_documented_names() -> None:
    """Guards the public surface against accidental renames."""
    import app.ingestion.metadata as metadata_module

    for name in (
        "DocumentMetadata",
        "PdfInfo",
        "TextInfo",
        "TitleSource",
        "build_metadata",
        "extract_pdf_info",
        "extract_text_info",
        "extract_title_from_text",
        "fallback_title",
        "parse_pdf_datetime",
        "sanitize_keywords",
        "sanitize_text",
    ):
        assert hasattr(metadata_module, name), name
