"""Tests for content-aware deduplication.

The module's whole reason to exist is telling two easily-confused cases
apart — a document that was **edited** (same path, new content) versus one
that was **renamed** (new path, same content) — because they demand
opposite handling: one replaces its stored chunks, the other must not create
a second copy of text already in the index. Most tests below therefore
assert on the *pair* of identifiers in a decision, not just its label,
because the label alone does not prove the caller can act correctly.

Hashing consistency gets equal attention: a digest only matches what it was
computed over, so a test proving ``str`` and ``bytes`` content agree is
really guarding against every file in the corpus reading as modified.
"""

from __future__ import annotations

import json

import pytest

from app.domain.errors import ValidationError
from app.domain.models import Document, content_sha256, make_doc_id
from app.ingestion.deduplication import (
    ChangeKind,
    DedupDecision,
    classify,
    find_by_doc_id,
    find_by_hash,
)


def make_doc(path: str, text: str, *, media_type: str = "text/markdown") -> Document:
    """Build a stored document the way the ingest pipeline will."""
    return Document.create(
        source_path=path,
        filename=path.rsplit("/", 1)[-1],
        media_type=media_type,
        content=text,
    )


def classify_source(known: list[Document], source_path: str, content: str | bytes):
    """Thin wrapper so every test reads as a question about deduplication."""
    return classify(known, source_path=source_path, content=content)


# --- NEW --------------------------------------------------------------------


def test_no_known_documents_means_new() -> None:
    decision = classify_source([], "notes/a.md", "some text")

    assert decision.kind is ChangeKind.NEW
    assert decision.is_new
    assert decision.existing_doc_id is None


def test_known_content_at_an_unknown_path_is_new() -> None:
    """Nothing at this path, and this content is not held elsewhere either."""
    decision = classify_source([make_doc("old.md", "different")], "new.md", "some text")

    assert decision.kind is ChangeKind.NEW


def test_new_decision_supplies_the_ids_a_caller_needs() -> None:
    """The caller is about to create a document, so it needs both ids."""
    decision = classify_source([], "notes/a.md", "some text")

    assert decision.doc_id == make_doc_id("notes/a.md")
    assert decision.content_sha256 == content_sha256("some text")


def test_new_needs_ingest_but_is_not_duplicate_content() -> None:
    decision = classify_source([], "notes/a.md", "some text")

    assert decision.needs_ingest
    assert not decision.is_duplicate_content


# --- UNCHANGED --------------------------------------------------------------


def test_same_path_and_same_content_is_unchanged() -> None:
    known = [make_doc("specs/api.md", "# API spec")]
    decision = classify_source(known, "specs/api.md", "# API spec")

    assert decision.kind is ChangeKind.UNCHANGED
    assert decision.is_unchanged


def test_unchanged_points_at_the_document_it_found() -> None:
    known = [make_doc("specs/api.md", "# API spec")]
    decision = classify_source(known, "specs/api.md", "# API spec")

    assert decision.existing_doc_id == decision.doc_id
    assert decision.existing_doc_id == known[0].doc_id


def test_unchanged_does_not_need_ingesting_but_is_duplicate_content() -> None:
    """An idempotent re-upload must be recognised as no work to do."""
    known = [make_doc("specs/api.md", "# API spec")]
    decision = classify_source(known, "specs/api.md", "# API spec")

    assert not decision.needs_ingest
    assert decision.is_duplicate_content


def test_unchanged_reason_says_nothing_to_do() -> None:
    known = [make_doc("specs/api.md", "# API spec")]
    decision = classify_source(known, "specs/api.md", "# API spec")

    assert "nothing to ingest" in decision.reason


# --- MODIFIED ---------------------------------------------------------------


def test_same_path_with_changed_content_is_modified() -> None:
    known = [make_doc("specs/api.md", "# API spec v1")]
    decision = classify_source(known, "specs/api.md", "# API spec v2")

    assert decision.kind is ChangeKind.MODIFIED
    assert decision.is_modified


def test_modified_keeps_the_same_doc_id_so_chunks_can_be_replaced() -> None:
    """THE contract: an edit keeps its id, so re-ingest overwrites old chunks.

    If the id changed on edit, stale chunks from the previous version would
    linger in the index and keep being retrieved.
    """
    known = [make_doc("specs/api.md", "# API spec v1")]
    decision = classify_source(known, "specs/api.md", "# API spec v2")

    assert decision.doc_id == known[0].doc_id
    assert decision.existing_doc_id == known[0].doc_id


def test_modified_changes_the_content_hash() -> None:
    """While the hash *does* change — that is how the edit is noticed."""
    known = [make_doc("specs/api.md", "# API spec v1")]
    decision = classify_source(known, "specs/api.md", "# API spec v2")

    assert decision.content_sha256 != known[0].content_sha256


def test_modified_needs_ingest_and_is_not_duplicate_content() -> None:
    known = [make_doc("specs/api.md", "# v1")]
    decision = classify_source(known, "specs/api.md", "# v2")

    assert decision.needs_ingest
    assert not decision.is_duplicate_content


def test_modified_reason_explains_replacement() -> None:
    known = [make_doc("specs/api.md", "# v1")]
    decision = classify_source(known, "specs/api.md", "# v2")

    assert "content changed" in decision.reason
    assert "replaces" in decision.reason


def test_document_created_with_a_precomputed_hash_still_classifies_unchanged() -> None:
    """The ``content_hash`` escape hatch must participate correctly.

    Documents built from file bytes (rather than from their extracted text)
    must still match content hashed the same way.
    """
    path = "specs/api.md"
    file_bytes = b"# API spec"
    stored = Document.create(
        source_path=path,
        filename="api.md",
        media_type="text/markdown",
        content="# API spec",
        content_hash=content_sha256(file_bytes),
    )

    decision = classify_source([stored], path, file_bytes)

    assert decision.kind is ChangeKind.UNCHANGED


# --- RENAMED ----------------------------------------------------------------


def test_new_path_with_known_content_is_renamed() -> None:
    known = [make_doc("specs/api.md", "# API spec")]
    decision = classify_source(known, "archive/api-copy.md", "# API spec")

    assert decision.kind is ChangeKind.RENAMED
    assert decision.is_renamed


def test_renamed_reports_both_ids_and_they_differ() -> None:
    """That difference is the entire point of detecting it."""
    known = [make_doc("specs/api.md", "# API spec")]
    decision = classify_source(known, "archive/api-copy.md", "# API spec")

    assert decision.doc_id == make_doc_id("archive/api-copy.md")
    assert decision.existing_doc_id == known[0].doc_id
    assert decision.existing_doc_id != decision.doc_id


def test_renamed_reason_names_the_document_already_holding_the_content() -> None:
    known = [make_doc("specs/api.md", "# API spec")]
    decision = classify_source(known, "archive/api-copy.md", "# API spec")

    assert known[0].doc_id in decision.reason


def test_renamed_is_duplicate_content_and_needs_no_ingest() -> None:
    known = [make_doc("specs/api.md", "# API spec")]
    decision = classify_source(known, "archive/api-copy.md", "# API spec")

    assert decision.is_duplicate_content
    assert not decision.needs_ingest


def test_rename_is_reported_not_resolved() -> None:
    """Content cannot tell a rename from a copy, so no policy is imposed.

    The decision reports the fact; the upload API decides what to do.
    """
    known = [make_doc("specs/api.md", "# API spec")]
    decision = classify_source(known, "backup/spec.md", "# API spec")

    assert decision.kind is ChangeKind.RENAMED
    # No "recommended action" is invented beyond the kind and the ids.
    assert decision.summary()["kind"] == "renamed"


# --- the path check outranks the content check ------------------------------


def test_an_edit_whose_new_bytes_match_another_document_is_still_modified() -> None:
    """Documented ordering: when the path is known, the path decides.

    Otherwise a file edited to equal a neighbour's content would be
    reclassified as a rename and this document's stale chunks would never be
    replaced.
    """
    doc_a = make_doc("specs/a.md", "content A")
    doc_b = make_doc("specs/b.md", "content B")

    decision = classify_source([doc_a, doc_b], "specs/a.md", "content B")

    assert decision.kind is ChangeKind.MODIFIED
    assert decision.existing_doc_id == doc_a.doc_id


def test_the_outranking_case_reports_the_path_owners_id_not_the_matches() -> None:
    doc_a = make_doc("specs/a.md", "content A")
    doc_b = make_doc("specs/b.md", "content B")

    decision = classify_source([doc_a, doc_b], "specs/a.md", "content B")

    assert decision.existing_doc_id != doc_b.doc_id


# --- caller argument errors propagate ---------------------------------------


def test_absolute_path_is_rejected() -> None:
    with pytest.raises(ValidationError) as excinfo:
        classify_source([], "/etc/passwd", "x")

    assert excinfo.value.code == "absolute_source_path"


def test_windows_absolute_path_is_rejected() -> None:
    with pytest.raises(ValidationError):
        classify_source([], "C:/secrets/notes.md", "x")


def test_path_traversal_is_rejected() -> None:
    with pytest.raises(ValidationError) as excinfo:
        classify_source([], "../outside/notes.md", "x")

    assert excinfo.value.code == "path_traversal"


def test_empty_path_is_rejected() -> None:
    """A bad path is a caller bug, not a file that happens to be new."""
    with pytest.raises(ValidationError) as excinfo:
        classify_source([], "", "x")

    assert excinfo.value.code == "empty_source_path"


def test_a_bad_path_is_never_reported_as_new() -> None:
    """The failure must not silently mint an id from a path that is wrong."""
    with pytest.raises(ValidationError):
        classify_source([make_doc("a.md", "x")], "/absolute/a.md", "x")


# --- hashing rules ----------------------------------------------------------


def test_str_and_bytes_content_hash_identically() -> None:
    """Guards the trap that would mark every file in a corpus as modified."""
    known = [make_doc("a.md", "hello world")]
    from_str = classify_source(known, "a.md", "hello world")
    from_bytes = classify_source(known, "a.md", b"hello world")

    assert from_str.content_sha256 == from_bytes.content_sha256
    assert from_str.kind is from_bytes.kind


def test_empty_content_hashes_to_the_sha256_of_nothing() -> None:
    decision = classify_source([], "a.md", "")

    assert decision.content_sha256 == content_sha256("")


def test_different_content_never_reports_unchanged() -> None:
    known = [make_doc("a.md", "one")]
    decision = classify_source(known, "a.md", "two")

    assert decision.kind is not ChangeKind.UNCHANGED


def test_non_ascii_content_hashes_stably() -> None:
    """Sinhala text must hash the same way regardless of how it arrives."""
    text = "Sinhala \u0dc3\u0dd2\u0d82\u0dc4\u0dba"

    assert content_sha256(text) == content_sha256(text.encode("utf-8"))


def test_whitespace_differences_change_the_hash() -> None:
    """A trailing newline is a real edit as far as bytes are concerned."""
    known = [make_doc("a.md", "line")]

    assert classify_source(known, "a.md", "line\n").kind is ChangeKind.MODIFIED


# --- lookup helpers ---------------------------------------------------------


def test_find_by_doc_id_returns_the_matching_document() -> None:
    doc = make_doc("specs/api.md", "# spec")

    assert find_by_doc_id([doc], doc.doc_id) is doc


def test_find_by_doc_id_returns_none_for_an_unknown_id() -> None:
    assert find_by_doc_id([make_doc("a.md", "x")], "doc_" + "0" * 16) is None


def test_find_by_hash_returns_the_matching_document() -> None:
    doc = make_doc("specs/api.md", "# spec")

    assert find_by_hash([doc], doc.content_sha256) is doc


def test_find_by_hash_returns_none_for_an_unknown_digest() -> None:
    assert find_by_hash([make_doc("a.md", "x")], "0" * 64) is None


def test_find_by_doc_id_does_not_match_by_hash() -> None:
    """The two lookups answer different questions and must not blur."""
    doc = make_doc("a.md", "x")

    assert find_by_doc_id([doc], doc.content_sha256) is None


def test_find_by_hash_does_not_match_by_doc_id() -> None:
    doc = make_doc("a.md", "x")

    assert find_by_hash([doc], doc.doc_id) is None


def test_find_by_hash_is_deterministic_when_content_repeats() -> None:
    """Two documents holding identical bytes: the first in order wins."""
    first = make_doc("a.md", "same")
    second = make_doc("b.md", "same")

    assert find_by_hash([first, second], content_sha256("same")) is first
    assert find_by_hash([second, first], content_sha256("same")) is second


def test_looking_up_in_an_empty_set_finds_nothing() -> None:
    assert find_by_doc_id([], "doc_" + "0" * 16) is None
    assert find_by_hash([], "0" * 64) is None


# --- flags, enum and summary ------------------------------------------------


def test_the_four_kinds_are_mutually_exclusive() -> None:
    cases = [
        ([], "a.md", "x"),
        ([make_doc("a.md", "x")], "a.md", "x"),
        ([make_doc("a.md", "x")], "a.md", "y"),
        ([make_doc("a.md", "x")], "b.md", "x"),
    ]
    for known, path, content in cases:
        d = classify_source(known, path, content)
        flags = (d.is_new, d.is_unchanged, d.is_modified, d.is_renamed)
        assert sum(flags) == 1, d


def test_needs_ingest_covers_exactly_new_and_modified() -> None:
    assert classify_source([], "a.md", "x").needs_ingest
    assert classify_source([make_doc("a.md", "x")], "a.md", "y").needs_ingest
    assert not classify_source([make_doc("a.md", "x")], "a.md", "x").needs_ingest
    assert not classify_source([make_doc("a.md", "x")], "b.md", "x").needs_ingest


def test_duplicate_content_covers_exactly_unchanged_and_renamed() -> None:
    assert classify_source([make_doc("a.md", "x")], "a.md", "x").is_duplicate_content
    assert classify_source([make_doc("a.md", "x")], "b.md", "x").is_duplicate_content
    assert not classify_source([], "a.md", "x").is_duplicate_content
    assert not classify_source([make_doc("a.md", "x")], "a.md", "y").is_duplicate_content


def test_change_kind_is_a_string_enum() -> None:
    """So a decision serialises straight to JSON without a custom encoder."""
    assert json.loads(json.dumps({"kind": ChangeKind.MODIFIED})) == {"kind": "modified"}
    assert classify_source([], "a.md", "x").kind == "new"
    assert {k.value for k in ChangeKind} == {"new", "unchanged", "modified", "renamed"}


def test_summary_is_json_safe() -> None:
    json.dumps(classify_source([], "a.md", "x").summary())  # must not raise


def test_summary_omits_the_content_hash() -> None:
    """Matches Document.summary: neither text nor its digest belongs in logs."""
    summary = classify_source([], "a.md", "x").summary()

    assert "content_sha256" not in summary


def test_summary_carries_the_ids_and_the_reason() -> None:
    known = [make_doc("a.md", "x")]
    summary = classify_source(known, "b.md", "x").summary()

    assert summary["kind"] == "renamed"
    assert summary["existing_doc_id"] == known[0].doc_id
    assert summary["doc_id"] == make_doc_id("b.md")
    assert summary["reason"]


def test_summary_reports_none_for_a_new_documents_existing_id() -> None:
    summary = classify_source([], "a.md", "x").summary()

    assert summary["existing_doc_id"] is None


def test_summary_contains_no_document_text() -> None:
    secret = "the confidential clause nine"
    summary = classify_source([make_doc("a.md", secret)], "b.md", secret).summary()

    assert secret not in json.dumps(summary)


# --- behaviour across a sequence --------------------------------------------


def test_the_same_file_classifies_unchanged_on_a_second_pass() -> None:
    """Idempotency: re-running the ingest of an untouched corpus does nothing."""
    known = [make_doc("a.md", "x")]
    decision = classify_source(known, "a.md", "x")

    assert decision.kind is ChangeKind.UNCHANGED
    assert not decision.needs_ingest


def test_decisions_are_deterministic_for_identical_input() -> None:
    known = [make_doc("a.md", "x")]
    first = classify_source(known, "b.md", "x")
    second = classify_source(known, "b.md", "x")

    assert first == second


def test_a_decision_is_immutable() -> None:
    """Frozen, so a caller cannot rewrite a verdict after acting on it."""
    import dataclasses

    decision = classify_source([], "a.md", "x")
    with pytest.raises(dataclasses.FrozenInstanceError):
        decision.kind = ChangeKind.NEW  # type: ignore[misc]


def test_module_exports_the_documented_names() -> None:
    import app.ingestion.deduplication as dedup_module

    for name in (
        "ChangeKind",
        "DedupDecision",
        "content_sha256",
        "find_by_doc_id",
        "find_by_hash",
        "make_doc_id",
    ):
        assert hasattr(dedup_module, name), name


def test_reexported_helpers_match_the_originals() -> None:
    """One import site for callers, but not a divergent second copy."""
    import app.ingestion.deduplication as dedup_module

    assert dedup_module.make_doc_id is make_doc_id
    assert dedup_module.content_sha256 is content_sha256


def test_decision_can_be_constructed_directly() -> None:
    """The dataclass is usable on its own for fakes and fixtures."""
    decision = DedupDecision(
        kind=ChangeKind.NEW,
        doc_id="doc_" + "0" * 16,
        content_sha256="0" * 64,
    )

    assert decision.is_new
    assert decision.existing_doc_id is None
