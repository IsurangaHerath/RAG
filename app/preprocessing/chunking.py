"""Turning a loaded document into the chunks that get embedded and retrieved.

Why chunks exist at all
-----------------------
An embedding model reads a whole passage at once, and a vector store ranks
passages, not files. Retrieving "this file" answers nothing when a question
is about *one paragraph* of it — and a citation that points at a 40-page PDF
is not a citation. So text is split into passages small enough to be
individually embedded, retrieved and quoted back as ``document : chunk``.

How the text is split
---------------------
Recursion, not a fixed stride: :class:`~langchain_text_splitters.RecursiveCharacterTextSplitter`
tries the most meaningful boundary first and only falls back to a shallower
one when the piece is still too big — headings, then blank lines, then
newlines, then spaces, then characters. A stride would cut sentences in
half and tables across rows; recursion keeps a passage readable, which is
what a model is graded on. Adjacent chunks overlap by ``chunk_overlap``
characters so a sentence spanning a boundary still appears on both sides.

Two structures are preserved rather than discarded on the way in:

* **Pages.** A PDF is split *page by page*. That costs the overlap between
  two pages, and buys an exact answer to "which page?" for every chunk —
  which is the whole reason section 5.2.3 kept pages instead of flattening them.
* **Sections.** Markdown is split at its ATX headings first, so each chunk
  can say which heading it sat under — as the full path from the document
  title down (``Endpoints > GET /payments/{id}``), because two sections
  elsewhere in one document can share a leaf name. The heading line itself
  stays in the chunk's text: a passage tagged "Rate limits" must still
  *contain* those words, or an embedding of it would be missing the very
  context the tag claims.

Splitting happens *inside* a unit, never across two, which means the
overlap does not cross a page or section boundary either. That is the
cost of exact attribution, and it is paid deliberately: a chunk that
straddled two pages would have to name one, and naming the wrong page is
worse for a citation than a missing sentence of context is.

Determinism
-----------
The same input and the same settings always produce the same chunks, same
order, same ids. Nothing here reads a clock or a random source, so
re-ingesting an untouched file overwrites the same ``chunk_id`` values
rather than appending a second copy of every chunk.

Deviation from the section 5.2 Protocol sketch
---------------------------------------
The protocol block in section 5.2 was written before Stages 11-12 and describes
loaders returning ``list[LCDocument]``; what was actually built (and what
section 5.2.2/section 5.2.3 document) is the native :class:`TextDocument` /
:class:`PdfDocument` pair. This module takes those directly, so the call
site the pipeline will use is ``chunker.split(loader.load(...), doc_id=...)``
with no conversion step in between — and LangChain's document type stays an
implementation detail inside this file rather than appearing in a signature
the rest of the project has to depend on.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Final

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from app.domain.errors import ValidationError
from app.domain.models import Chunk
from app.ingestion.loaders import TextDocument
from app.ingestion.pdf_loader import PdfDocument

__all__ = [
    "DEFAULT_CHUNK_OVERLAP",
    "DEFAULT_CHUNK_SIZE",
    "RecursiveChunker",
]

#: Size the project ships with. Kept in step with ``Settings.chunk_size`` by
#: a test, so the two defaults cannot drift into disagreeing about what a
#: chunk is.
DEFAULT_CHUNK_SIZE: Final = 1000

#: Overlap the project ships with; see ``Settings.chunk_overlap``.
DEFAULT_CHUNK_OVERLAP: Final = 150

#: An ATX heading: ``# Title`` through ``###### Title``, with optional
#: closing hashes. Group 1 is the level (its hash count), group 2 the
#: title. Setext underlines (``Title\\n=====``) are not matched: a single
#: ``=`` under a line is indistinguishable from a horizontal rule in
#: arbitrary text, and guessing wrong would invent a section boundary.
_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")

#: Joins ancestor titles into one section path, e.g.
#: ``"Endpoints > GET /payments/{id}"``.
_SECTION_SEPARATOR = " > "

#: Opening or closing of a fenced code block (``` or ~~~).
_FENCE_RE = re.compile(r"^[ \t]*(?:```|~~~)")


class RecursiveChunker:
    """Split a loaded document into overlapping, size-bounded chunks.

    The implementation of section 5.2's ``Chunker`` interface, named after the
    splitting strategy rather than the interface itself so a second
    strategy (page-per-chunk for citations, say) could sit beside it
    without either being called "the" chunker.

    Parameters
    ----------
    chunk_size:
        Target maximum characters per chunk. Must be at least 1. One
        documented exception: a heading glued to the first paragraph of its
        own section may overshoot by the heading's length — see
        :func:`_fold_leading_heading`, which explains why that is the better
        trade. Every other chunk stays within the limit.
    chunk_overlap:
        Characters repeated at the end of a chunk and the start of the
        next, so text straddling a boundary is still present whole in one
        of them. Must be at least 0 and strictly smaller than
        ``chunk_size`` — overlap the size of the chunk would make the
        splitter recurse forever.

    Raises
    ------
    ValidationError
        A setting is out of range. Raised in the constructor rather than
        partway through a document, so a bad configuration fails on the
        line that set it instead of after doing half the work.

    Notes
    -----
    Instances hold no per-document state, so one chunker can be shared
    across a whole batch.
    """

    def __init__(
        self,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        _validate_split_settings(chunk_size, chunk_overlap)
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

        # Built once, not per call: two splitters because markdown earns a
        # different separator list (headings and fences first) than prose.
        self._plain_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self._markdown_splitter = RecursiveCharacterTextSplitter.from_language(
            Language.MARKDOWN,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    # -- configuration ------------------------------------------------------

    @property
    def chunk_size(self) -> int:
        """Target maximum characters per chunk (see the class docstring)."""
        return self._chunk_size

    @property
    def chunk_overlap(self) -> int:
        """Characters shared by adjacent chunks."""
        return self._chunk_overlap

    # -- splitting ----------------------------------------------------------

    def split(
        self,
        document: TextDocument | PdfDocument,
        *,
        doc_id: str,
    ) -> list[Chunk]:
        """Split ``document`` into chunks numbered from zero.

        Parameters
        ----------
        document:
            What a loader returned. Pages, sections and headings are read
            from it rather than re-derived, because they only exist at
            this point in the pipeline.
        doc_id:
            The id every chunk will be stamped with. Validated by the
            first chunk that is built.

        Returns
        -------
        list[Chunk]
            In source order, with ``chunk_id = f"{doc_id}:{index}"``.
            Empty when the source has no text — an empty file or a scan
            with no text layer — which a caller indexing a mixed batch
            expects rather than treats as an error.

        Notes
        -----
        A chunk never spans two pages: it would have to name one, and
        naming the wrong one is worse than losing the overlap.
        """
        splitter = self._splitter_for(document)
        base = _base_metadata(document)

        chunks: list[Chunk] = []
        for text, provenance in self._units(document):
            # No blank-piece guard here: the splitter strips whitespace, so
            # every piece it returns carries text, and if that ever stopped
            # being true ``Chunk.create`` rejects it with ``empty_content``
            # rather than letting a silent skip hide a library change.
            for piece in _fold_leading_heading(text, splitter.split_text(text)):
                chunks.append(
                    Chunk.create(
                        doc_id=doc_id,
                        index=len(chunks),
                        content=piece,
                        metadata={**base, **provenance},
                    )
                )
        return chunks

    # -- internals ----------------------------------------------------------

    def _units(
        self,
        document: TextDocument | PdfDocument,
    ) -> Iterator[tuple[str, dict[str, str | int | float | bool]]]:
        """Yield each run of text to split, with the provenance it carries.

        A *unit* is the largest span that must not be mixed with any other:
        a page for PDFs (so ``page`` is exact), a heading section for
        Markdown (so ``section`` is exact), and the whole file for plain
        text. Splitting happens inside a unit, never across two.
        """
        if isinstance(document, PdfDocument):
            for page in document.pages:
                if page.text.strip():
                    yield page.text, {"page": page.page_number}
            return

        if not document.text.strip():
            return

        if document.media_type == "text/markdown":
            for section, section_text in _markdown_sections(document.text):
                if section_text.strip():
                    # A preamble before the first heading has no section;
                    # leaving the key out is honest, where section="" would
                    # read as "a section called nothing".
                    provenance: dict[str, str | int | float | bool] = (
                        {"section": section} if section else {}
                    )
                    yield section_text, provenance
            return

        yield document.text, {}

    def _splitter_for(self, document: TextDocument | PdfDocument) -> RecursiveCharacterTextSplitter:
        """Pick the separator list that suits this source.

        Markdown gets headings and code fences as first-choice boundaries;
        everything else — prose and extracted PDF text alike — gets the
        generic blank-line-first recursion.
        """
        if isinstance(document, TextDocument) and document.media_type == "text/markdown":
            return self._markdown_splitter
        return self._plain_splitter


def _validate_split_settings(chunk_size: int, chunk_overlap: int) -> None:
    """Reject settings the underlying library would refuse as a bare ValueError.

    ``langchain`` raises ``ValueError`` for both cases; ours is a
    :class:`RAGError` with a stable code, so a caller can distinguish "the
    configuration is wrong" from "this document is wrong" — the same
    distinction the ingest report and the CLI keep everywhere else.
    """
    if chunk_size < 1:
        raise ValidationError(
            f"chunk_size must be >= 1, got {chunk_size}",
            code="invalid_chunk_size",
        )
    if chunk_overlap < 0:
        raise ValidationError(
            f"chunk_overlap must be >= 0, got {chunk_overlap}",
            code="invalid_chunk_overlap",
        )
    if chunk_overlap >= chunk_size:
        raise ValidationError(
            f"chunk_overlap ({chunk_overlap}) must be smaller than chunk_size ({chunk_size})",
            code="invalid_chunk_overlap",
        )


def _base_metadata(
    document: TextDocument | PdfDocument,
) -> dict[str, str | int | float | bool]:
    """Provenance every chunk inherits, whatever the source.

    These two keys exist because section 5.3's storage schema lists them as chunk
    metadata; attaching them here means the indexing stage copies what it
    is given instead of reconstructing it from elsewhere.
    """
    return {"filename": document.source_name, "media_type": document.media_type}


def _fold_leading_heading(text: str, pieces: list[str]) -> list[str]:
    """Glue a lone leading heading to the first paragraph of its own section.

    A heading sits above a blank line, and a blank line is the first place
    the splitter looks — so a section larger than ``chunk_size`` would
    otherwise open with a heading-only chunk: a dozen characters that can be
    retrieved and answers nothing, while the paragraph it introduces is
    embedded with no idea which heading it belongs to.

    **This is the one place a chunk may exceed ``chunk_size``**, by exactly
    the length of its heading line. That is a deliberate trade: the size is
    a quality target rather than a hard ceiling (embeddings accept far more
    than 1000 characters, and section 5.4 counts tokens when assembling a prompt),
    and a paragraph wearing its own heading is worth more than the bound is.
    A chunk with no heading to glue never overshoots.

    The result is re-sliced from ``text`` rather than built by
    concatenation. The splitter drops the separator between the two pieces,
    so ``pieces[0] + pieces[1]`` would produce text that appears nowhere in
    the document — invented content, which is exactly what a chunk must
    never be.

    Only the *first* piece is considered, because that is where this
    module's own section headings land; a heading in the middle of a body
    belongs to the splitter's judgement, not this rule's.
    """
    if len(pieces) < 2 or not pieces[1].strip():
        return pieces
    if not _HEADING_RE.match(pieces[0].strip()):
        return pieces

    start = text.find(pieces[1], len(pieces[0]))
    if start == -1:  # defensive: pieces come from this text, so unreachable
        return pieces

    end = start + len(pieces[1])
    return [text[:end], *pieces[2:]]


def _markdown_sections(text: str) -> list[tuple[str | None, str]]:
    """Split Markdown at ATX headings into sections with full heading paths.

    Returns ``(section, section_text)`` pairs in order. ``section`` is the
    *breadcrumb* of every heading above the text — ``"Endpoints > GET
    /payments/{id}"`` — because a bare leaf title cannot tell two sections
    that share a name apart, and the path is the thing a citation can
    actually be checked against. Text before the first heading has
    ``section is None``.

    The heading line stays *inside* its section rather than being peeled
    off into metadata: dropping it would leave the chunk's text missing the
    words its own ``section`` tag refers to.

    **Empty sections fold into their first descendant.** A heading with no
    body of its own — because the very next line is another heading — would
    otherwise become a section whose entire text is the heading, i.e. a
    chunk of a dozen characters that can be retrieved and answers nothing.
    Instead its heading line is prepended to the child's text, where the
    child's breadcrumb already contains the parent's title, so the label
    and the text still agree. A heading with no descendant to fold into
    keeps its own section rather than losing its words.

    Headings inside fenced code blocks are ignored — ``# not a real heading``
    appears in every shell snippet, and splitting there would chop a script
    into pieces labelled after its own comments.
    """
    preamble: list[str] = []
    # (breadcrumb, heading level, lines — the heading line first)
    headings: list[tuple[str, int, list[str]]] = []
    in_fence = False
    stack: list[str] = []
    stack_levels: list[int] = []

    def append_to_current(line: str) -> None:
        if headings:
            headings[-1][2].append(line)
        else:
            preamble.append(line)

    for line in text.splitlines(keepends=True):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            append_to_current(line)
            continue

        match = None if in_fence else _HEADING_RE.match(line.rstrip("\r\n"))
        if match is None:
            append_to_current(line)
            continue

        # A heading closes every open heading at its own level or deeper,
        # which is what makes the stack the current ancestor path.
        level = len(match.group(1))
        while stack_levels and stack_levels[-1] >= level:
            stack.pop()
            stack_levels.pop()
        stack.append(match.group(2))
        stack_levels.append(level)
        headings.append((_SECTION_SEPARATOR.join(stack), level, [line]))

    sections: list[tuple[str | None, str]] = []
    if preamble:
        sections.append((None, "".join(preamble)))
    sections.extend(_merge_empty_sections(headings))
    return sections


def _merge_empty_sections(headings: list[tuple[str, int, list[str]]]) -> list[tuple[str, str]]:
    """Fold each body-less heading into the first descendant that follows.

    Walking backwards lets a run of empty ancestors (``## A`` with only
    ``### B`` under it, itself empty) collapse in one pass: each one
    prepends its heading line to the entry already carrying its
    descendants' text. The surviving entry keeps its *child's* breadcrumb —
    the parent's title is in that breadcrumb already, and the parent's
    heading line is now the first thing the text says.
    """
    merged: list[tuple[str, list[str]]] = []

    for breadcrumb, _level, lines in reversed(headings):
        is_body_less = not any(line.strip() for line in lines[1:])
        first_is_descendant = merged and merged[-1][0].startswith(breadcrumb + _SECTION_SEPARATOR)
        if is_body_less and first_is_descendant:
            merged[-1] = (merged[-1][0], [*lines, *merged[-1][1]])
        else:
            merged.append((breadcrumb, lines))

    merged.reverse()
    return [(section, "".join(lines)) for section, lines in merged]
