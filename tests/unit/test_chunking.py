"""Tests for chunking: the ``Chunk`` record (§5.1) and the chunker (§5.2).

The properties that matter, in the order they matter:

1. **A chunk is a piece of the real document.** Not a near-copy, not a join
   with a missing separator — a substring. What a user is quoted back must
   be literally what was written, and a join built by concatenation would
   silently invent text that appears nowhere in the source.
2. **Every chunk can be named and located.** ``doc_id:index`` for identity,
   plus the page or section it came from, because a citation a reader
   cannot check is not a citation.
3. **The same input produces the same chunks.** Otherwise re-ingesting an
   unchanged file would append a second copy of everything.

Tests build their inputs with the real loaders rather than hand-made
model objects: the chunker's contract is with what the loaders return, so
a test that fabricated the input would never notice the two drifting apart.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings
from app.domain.errors import ValidationError
from app.domain.models import Chunk
from app.ingestion.loaders import TextDocument, load_text
from app.ingestion.pdf_loader import PdfDocument, load_pdf
from app.preprocessing.chunking import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    RecursiveChunker,
    _fold_leading_heading,
)
from scripts.pdf_fixtures import build_blank_pdf, build_pdf

DOC_ID = "doc_0123456789abcdef"

LONG_PROSE = (
    "Technical documents are read in pieces. "
    * 6
    + "An embedding model sees one passage at a time, so the passage has to "
    "stand on its own. " * 4
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _text_document(
    tmp_path: Path,
    body: str,
    *,
    name: str = "notes.md",
) -> TextDocument:
    """Write ``body`` and load it back through the real text loader."""
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return load_text(path)


def _pdf_document(tmp_path: Path, pages: list[str]) -> PdfDocument:
    """Build a real PDF with one page of text per entry, and load it."""
    path = tmp_path / "sample.pdf"
    path.write_bytes(build_pdf(pages))
    return load_pdf(path)


def _sections_of(chunks: list[Chunk]) -> set[str]:
    """Every section named by any chunk."""
    return {c.metadata["section"] for c in chunks if "section" in c.metadata}


def _overlap_length(a: str, b: str) -> int:
    """Length of the longest suffix of ``a`` that starts ``b``."""
    for size in range(min(len(a), len(b)), 0, -1):
        if a.endswith(b[:size]):
            return size
    return 0


# ---------------------------------------------------------------------------
# Chunk record
# ---------------------------------------------------------------------------
def test_create_derives_the_chunk_id() -> None:
    """The id is a position, so callers cannot supply one that disagrees."""
    chunk = Chunk.create(doc_id=DOC_ID, index=7, content="hello")

    assert chunk.chunk_id == f"{DOC_ID}:7"
    assert chunk.doc_id == DOC_ID
    assert chunk.index == 7


def test_create_copies_metadata_so_later_mutation_cannot_reach_a_chunk() -> None:
    """A chunk handed to a renderer must not change when the caller does."""
    metadata = {"page": 3}
    chunk = Chunk.create(doc_id=DOC_ID, index=0, content="x", metadata=metadata)

    metadata["page"] = 99

    assert chunk.metadata == {"page": 3}


def test_metadata_defaults_to_empty() -> None:
    """Chunks from an unstructured source carry nothing extra, not ``None``."""
    assert Chunk.create(doc_id=DOC_ID, index=0, content="x").metadata == {}


def test_a_chunk_id_that_disagrees_with_its_position_is_rejected() -> None:
    """A wrong id would silently break idempotent re-ingestion."""
    with pytest.raises(ValidationError) as excinfo:
        Chunk(
            chunk_id=f"{DOC_ID}:99",
            doc_id=DOC_ID,
            index=0,
            content="x",
        )

    assert excinfo.value.code == "invalid_chunk_id"


def test_a_negative_index_is_rejected() -> None:
    """Negative positions cannot occur in a document, so they cannot be built."""
    with pytest.raises(ValidationError) as excinfo:
        Chunk.create(doc_id=DOC_ID, index=-1, content="x")

    assert excinfo.value.code == "negative_chunk_index"


def test_a_malformed_doc_id_is_rejected() -> None:
    """Chunks inherit the document id's contract, they do not relax it."""
    with pytest.raises(ValidationError) as excinfo:
        Chunk.create(doc_id="not-an-id", index=0, content="x")

    assert excinfo.value.code == "invalid_doc_id"


def test_empty_content_is_rejected() -> None:
    """An empty chunk would be embedded as nothing and retrieved as nothing."""
    with pytest.raises(ValidationError) as excinfo:
        Chunk.create(doc_id=DOC_ID, index=0, content="   ")

    assert excinfo.value.code == "empty_content"


def test_metadata_must_hold_scalars_only() -> None:
    """The same rule as document metadata: Chroma and JSON cannot store a graph."""
    with pytest.raises(ValidationError) as excinfo:
        Chunk.create(doc_id=DOC_ID, index=0, content="x", metadata={"page": {"n": 1}})  # type: ignore[dict-item]

    assert excinfo.value.code == "invalid_metadata_value"


def test_summary_excludes_the_text() -> None:
    """Chunk text is what a user gets quoted, so it must never be summarised."""
    chunk = Chunk.create(
        doc_id=DOC_ID,
        index=3,
        content="secret passage",
        metadata={"page": 2},
    )

    summary = chunk.summary()

    assert "secret passage" not in str(summary)
    assert summary["chunk_id"] == f"{DOC_ID}:3"
    assert summary["char_count"] == len("secret passage")
    assert summary["metadata"] == {"page": 2}


def test_chunks_are_immutable() -> None:
    """One task must not be able to rewrite a chunk another is reading."""
    chunk = Chunk.create(doc_id=DOC_ID, index=0, content="x")

    with pytest.raises((AttributeError, TypeError)):
        chunk.content = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Settings drift
# ---------------------------------------------------------------------------
def test_default_settings_match_the_configuration() -> None:
    """The module defaults and ``Settings`` must agree on what a chunk is.

    Two copies of 1000/150 in different files would drift the first time
    either was edited, and the discrepancy would only ever show up as
    slightly worse retrieval.
    """
    settings = Settings(_env_file=None)

    assert DEFAULT_CHUNK_SIZE == settings.chunk_size
    assert DEFAULT_CHUNK_OVERLAP == settings.chunk_overlap
    assert RecursiveChunker().chunk_size == settings.chunk_size
    assert RecursiveChunker().chunk_overlap == settings.chunk_overlap


def test_invalid_split_settings_are_rejected_at_construction() -> None:
    """A bad size fails on the line that set it, not partway through a file."""
    with pytest.raises(ValidationError) as too_small:
        RecursiveChunker(chunk_size=0)

    with pytest.raises(ValidationError) as negative_overlap:
        RecursiveChunker(chunk_overlap=-1)

    with pytest.raises(ValidationError) as overlap_too_large:
        RecursiveChunker(chunk_size=100, chunk_overlap=100)

    assert too_small.value.code == "invalid_chunk_size"
    assert negative_overlap.value.code == "invalid_chunk_overlap"
    assert overlap_too_large.value.code == "invalid_chunk_overlap"


# ---------------------------------------------------------------------------
# Plain text
# ---------------------------------------------------------------------------
def test_text_shorter_than_the_limit_becomes_one_chunk(tmp_path: Path) -> None:
    """The simplest case: the document, kept whole, named as the first chunk."""
    document = _text_document(tmp_path, "A short note.", name="notes.txt")

    chunks = RecursiveChunker().split(document, doc_id=DOC_ID)

    assert len(chunks) == 1
    assert chunks[0].content == "A short note."
    assert chunks[0].chunk_id == f"{DOC_ID}:0"
    assert chunks[0].metadata["filename"] == "notes.txt"
    assert chunks[0].metadata["media_type"] == "text/plain"


def test_long_text_is_split_within_the_size_limit(tmp_path: Path) -> None:
    """No chunk grows past the limit the setting promises."""
    document = _text_document(tmp_path, LONG_PROSE, name="notes.txt")
    chunker = RecursiveChunker(chunk_size=120, chunk_overlap=30)

    chunks = chunker.split(document, doc_id=DOC_ID)

    assert len(chunks) > 1
    assert all(chunk.char_count <= 120 for chunk in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert [c.chunk_id for c in chunks] == [f"{DOC_ID}:{i}" for i in range(len(chunks))]


def test_adjacent_chunks_overlap_so_a_boundary_is_not_a_seam(tmp_path: Path) -> None:
    """Text spanning a cut is present whole on at least one side of it."""
    document = _text_document(tmp_path, LONG_PROSE, name="notes.txt")
    chunks = RecursiveChunker(chunk_size=120, chunk_overlap=30).split(document, doc_id=DOC_ID)

    assert (
        max(
            _overlap_length(chunks[i].content, chunks[i + 1].content)
            for i in range(len(chunks) - 1)
        )
        > 0
    )


def test_every_chunk_is_a_substring_of_the_document(tmp_path: Path) -> None:
    """The strongest claim here: chunks are cut from the text, never rebuilt.

    A join that dropped a separator — ``"rate"`` + ``"limits"`` producing
    ``"ratelimits"`` — would embed text the author never wrote, and nothing
    downstream would ever catch it.
    """
    document = _text_document(tmp_path, LONG_PROSE, name="notes.txt")
    chunks = RecursiveChunker(chunk_size=120, chunk_overlap=30).split(document, doc_id=DOC_ID)

    assert all(chunk.content in document.text for chunk in chunks)


def test_splitting_twice_returns_identical_chunks(tmp_path: Path) -> None:
    """Determinism, which is what makes re-ingestion replace rather than add."""
    document = _text_document(tmp_path, LONG_PROSE, name="notes.txt")
    chunker = RecursiveChunker(chunk_size=120, chunk_overlap=30)

    first = chunker.split(document, doc_id=DOC_ID)
    second = chunker.split(document, doc_id=DOC_ID)

    assert [(c.chunk_id, c.content, dict(c.metadata)) for c in first] == [
        (c.chunk_id, c.content, dict(c.metadata)) for c in second
    ]


def test_one_chunker_serves_several_documents(tmp_path: Path) -> None:
    """Instances hold no per-document state, so a batch can share one."""
    chunker = RecursiveChunker(chunk_size=64, chunk_overlap=10)
    first = _text_document(tmp_path, LONG_PROSE, name="one.txt")
    second = _text_document(tmp_path, "Something else entirely.", name="two.txt")

    from_first = chunker.split(first, doc_id=DOC_ID)
    from_second = chunker.split(second, doc_id="doc_fedcba9876543210")

    assert from_first[0].chunk_id.startswith(f"{DOC_ID}:")
    assert from_second[0].chunk_id.startswith("doc_fedcba9876543210:")
    assert from_second[0].metadata["filename"] == "two.txt"


@pytest.mark.parametrize("body", ["", "   \n\n\t  "])
def test_text_without_content_yields_no_chunks(tmp_path: Path, body: str) -> None:
    """Empty is a case to handle, not an error — loaders can return it."""
    document = _text_document(tmp_path, body, name="empty.txt")

    assert RecursiveChunker().split(document, doc_id=DOC_ID) == []


# ---------------------------------------------------------------------------
# Markdown sections
# ---------------------------------------------------------------------------
def test_markdown_chunks_carry_the_heading_they_sit_under(tmp_path: Path) -> None:
    """Section attribution, which is half of what makes a citation checkable."""
    body = "# Guide\n\nIntro paragraph.\n\n## Rate limits\n\nBody text here.\n"
    document = _text_document(tmp_path, body, name="guide.md")

    chunks = RecursiveChunker().split(document, doc_id=DOC_ID)

    assert _sections_of(chunks) == {"Guide", "Guide > Rate limits"}
    assert all(c.metadata["media_type"] == "text/markdown" for c in chunks)


def test_text_before_the_first_heading_has_no_section(tmp_path: Path) -> None:
    """Absence is honest; ``section=""`` would read as a section named nothing."""
    body = "Preamble line.\n\n# Heading\n\nBody.\n"
    document = _text_document(tmp_path, body, name="guide.md")

    chunks = RecursiveChunker().split(document, doc_id=DOC_ID)

    assert any("section" not in c.metadata for c in chunks)
    assert _sections_of(chunks) == {"Heading"}


def test_a_whitespace_only_preamble_contributes_nothing(tmp_path: Path) -> None:
    """Leading blank lines are a gap, not a section worth embedding.

    Without the skip they would become a chunk of pure whitespace — useless
    to embed and noise to retrieve.
    """
    body = "   \n\n\t\n# Heading\n\nBody text.\n"
    document = _text_document(tmp_path, body, name="guide.md")

    chunks = RecursiveChunker().split(document, doc_id=DOC_ID)

    assert chunks  # the heading section still split normally
    assert all(c.content.strip() for c in chunks)
    assert _sections_of(chunks) == {"Heading"}


def test_the_heading_stays_inside_the_chunks_of_its_own_section(
    tmp_path: Path,
) -> None:
    """A chunk tagged "Rate limits" must still contain the words it is tagged with.

    Otherwise an embedding of that chunk is missing the very context the
    tag claims, and a reader shown the chunk sees no heading at all.
    """
    body = "## Rate limits\n\nEach account may authorise payments.\n"
    document = _text_document(tmp_path, body, name="guide.md")

    chunks = RecursiveChunker().split(document, doc_id=DOC_ID)

    tagged = [c for c in chunks if c.metadata.get("section") == "Rate limits"]
    assert tagged
    assert any("## Rate limits" in c.content for c in tagged)


def test_a_large_section_does_not_open_with_a_heading_alone(tmp_path: Path) -> None:
    """The fold: a section bigger than the limit must not strand its heading.

    Left alone, the split would produce a fourteen-character chunk that can
    be retrieved and answers nothing.
    """
    body = "## Rate limits\n\n" + ("Each account may authorise payments. " * 30)
    document = _text_document(tmp_path, body, name="guide.md")

    chunks = RecursiveChunker(chunk_size=200, chunk_overlap=40).split(document, doc_id=DOC_ID)

    assert len(chunks) > 1
    assert all(c.content.strip() != "## Rate limits" for c in chunks)
    assert all(c.content in document.text for c in chunks)
    # The only chunks allowed past the limit are the folded one, and only
    # by as much as its own heading line.
    assert all(c.char_count <= 200 + len("## Rate limits") for c in chunks)


def test_a_heading_inside_a_fence_is_not_treated_as_a_section(
    tmp_path: Path,
) -> None:
    """``# comment`` appears in every shell snippet; splitting there would cut
    a script into pieces labelled after its own comments."""
    body = (
        "# Guide\n\n"
        "Intro.\n"
        "```shell\n"
        "# this is a comment, not a heading\n"
        "echo hi\n"
        "```\n"
        "\n"
        "## After the fence\n\n"
        "Closing.\n"
    )
    document = _text_document(tmp_path, body, name="guide.md")

    chunks = RecursiveChunker().split(document, doc_id=DOC_ID)

    assert _sections_of(chunks) == {"Guide", "Guide > After the fence"}
    assert any("# this is a comment" in c.content for c in chunks)


def test_sections_are_full_paths_rather_than_bare_leaf_titles(tmp_path: Path) -> None:
    """A leaf name alone cannot tell two identically-named sections apart.

    The path is also what a reader can follow back to the document, which
    is the point of attaching a section to a chunk at all.
    """
    body = "# API\n\nIntro.\n\n## Errors\n\nBody.\n\n### Rate limits\n\nDetails.\n"
    document = _text_document(tmp_path, body, name="guide.md")

    chunks = RecursiveChunker().split(document, doc_id=DOC_ID)

    assert _sections_of(chunks) == {"API", "API > Errors", "API > Errors > Rate limits"}


def test_a_sibling_heading_does_not_inherit_its_predecessor(tmp_path: Path) -> None:
    """Closing an open heading is what keeps two siblings apart.

    Without popping, the second chapter would be filed under the first
    chapter's name — every later section attached to the wrong parent.
    """
    body = (
        "# API\n\nIntro.\n\n"
        "## Errors\n\nBody.\n\n"
        "## Rate limits\n\nBody.\n\n"
        "# Other document\n\nBody.\n"
    )
    document = _text_document(tmp_path, body, name="guide.md")

    chunks = RecursiveChunker().split(document, doc_id=DOC_ID)

    assert _sections_of(chunks) == {
        "API",
        "API > Errors",
        "API > Rate limits",
        "Other document",
    }


def test_an_empty_heading_folds_into_its_first_descendant(tmp_path: Path) -> None:
    """A heading whose body is only the next heading must not become a chunk.

    That is the twelve-character ``## Endpoints`` the real sample corpus
    produces: retrievable, answering nothing, while every endpoint below it
    is embedded with no idea which heading it belongs to. The heading line
    moves into the child's text, where the child's path already names it —
    so the label and the text keep agreeing.
    """
    body = "## Endpoints\n### GET /payments/{id}\n\nReturns one payment.\n"
    document = _text_document(tmp_path, body, name="api.md")

    chunks = RecursiveChunker().split(document, doc_id=DOC_ID)

    assert not any(c.content.strip() == "## Endpoints" for c in chunks)
    assert _sections_of(chunks) == {"Endpoints > GET /payments/{id}"}
    assert any(c.content.startswith("## Endpoints") for c in chunks)
    assert all(c.content in document.text for c in chunks)


def test_an_empty_heading_with_no_descendant_keeps_its_own_section(
    tmp_path: Path,
) -> None:
    """Nowhere to fold to, so the words stay where they are written.

    Dropping the section would delete the heading from the index entirely;
    folding it backwards would file it under a heading that does not
    contain it.
    """
    body = "# Guide\n\nIntro.\n\n## Orphan\n"
    document = _text_document(tmp_path, body, name="guide.md")

    chunks = RecursiveChunker().split(document, doc_id=DOC_ID)

    assert _sections_of(chunks) == {"Guide", "Guide > Orphan"}
    assert any(c.content.strip() == "## Orphan" for c in chunks)


# ---------------------------------------------------------------------------
# PDFs and pages
# ---------------------------------------------------------------------------
def test_every_pdf_chunk_names_the_page_it_came_from(tmp_path: Path) -> None:
    """The positional guarantee §5.2.3 exists for."""
    pages = [
        "Chapter one begins here. " + "Sentence. " * 30,
        "Chapter two starts here. " + "Sentence. " * 30,
        "Chapter three ends here.",
    ]
    document = _pdf_document(tmp_path, pages)
    chunks = RecursiveChunker(chunk_size=200, chunk_overlap=40).split(document, doc_id=DOC_ID)

    assert [c.metadata["page"] for c in chunks] == sorted(c.metadata["page"] for c in chunks)
    assert {c.metadata["page"] for c in chunks} == {1, 2, 3}
    # Each chunk is text from its own page and no other.
    assert all(c.content in pages[c.metadata["page"] - 1] for c in chunks)


def test_a_page_longer_than_the_limit_still_produces_several_chunks(
    tmp_path: Path,
) -> None:
    """Splitting inside a page keeps the page number on every piece."""
    pages = ["First page text. " + "Filler sentence goes here. " * 40]
    document = _pdf_document(tmp_path, pages)

    chunks = RecursiveChunker(chunk_size=150, chunk_overlap=30).split(document, doc_id=DOC_ID)

    assert len(chunks) > 1
    assert {c.metadata["page"] for c in chunks} == {1}
    assert all(c.char_count <= 150 for c in chunks)


def test_a_pdf_without_a_text_layer_yields_no_chunks(tmp_path: Path) -> None:
    """A scan is not an error; §5.2.5 marks it skipped before it gets here."""
    path = tmp_path / "scan.pdf"
    path.write_bytes(build_blank_pdf(2))
    document = load_pdf(path)

    assert RecursiveChunker().split(document, doc_id=DOC_ID) == []


def test_pdf_chunks_carry_filename_and_media_type(tmp_path: Path) -> None:
    """The two keys §5.3 lists for every chunk, whatever the source."""
    document = _pdf_document(tmp_path, ["Some page text."])

    chunks = RecursiveChunker().split(document, doc_id=DOC_ID)

    assert chunks[0].metadata["filename"] == "sample.pdf"
    assert chunks[0].metadata["media_type"] == "application/pdf"
    assert "section" not in chunks[0].metadata  # PDFs have no reliable headings


# ---------------------------------------------------------------------------
# The heading fold, in isolation
# ---------------------------------------------------------------------------
def test_fold_glues_a_lone_heading_to_its_paragraph() -> None:
    text = "## Heading\n\nBody text."
    pieces = ["## Heading", "Body text."]

    assert _fold_leading_heading(text, pieces) == ["## Heading\n\nBody text."]


def test_fold_glues_even_when_the_pair_outgrows_the_chunk_size() -> None:
    """The size bound yields to the heading, deliberately.

    A heading-only chunk can be retrieved and answers nothing, while the
    paragraph it introduces is embedded without knowing its own heading.
    ``chunk_size`` is a quality target, not a ceiling anything downstream
    depends on, so the trade favours the fold every time.
    """
    text = "## Heading\n\n" + "A " * 200  # far longer than any sane chunk
    pieces = ["## Heading", text[len("## Heading\n\n") :]]

    folded = _fold_leading_heading(text, pieces)

    assert folded[0].startswith("## Heading\n\n")
    assert len(folded) == 1  # the two pieces became one, size notwithstanding


def test_fold_does_nothing_without_a_leading_heading() -> None:
    text = "Just prose.\n\nMore prose."
    pieces = ["Just prose.", "More prose."]

    assert _fold_leading_heading(text, pieces) == pieces


def test_fold_does_nothing_when_the_heading_is_the_whole_section() -> None:
    """Nothing to fold into, and dropping the text would lose content."""
    text = "## Heading\n"

    assert _fold_leading_heading(text, ["## Heading"]) == ["## Heading"]


def test_fold_does_nothing_when_the_next_piece_is_only_whitespace() -> None:
    """Defensive: glue to words, never to a gap."""
    text = "## Heading\n\n   \nBody"
    pieces = ["## Heading", "   "]

    assert _fold_leading_heading(text, pieces) == pieces


def test_fold_falls_back_to_the_pieces_if_the_text_does_not_contain_them() -> None:
    """Defensive: a mismatch must return the split unchanged, never fabricate."""
    assert _fold_leading_heading("other text", ["## H", "body"]) == [
        "## H",
        "body",
    ]
