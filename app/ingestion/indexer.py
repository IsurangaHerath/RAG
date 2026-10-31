"""Batch indexing: the pass that turns a directory into indexed documents (Stage 28).

section 5.16's ``IngestPipeline`` indexes *one* document when someone hands it a
file. Nothing yet walks a whole directory, owns the registry row a document
is supposed to have while work is happening (section 6.1), and reports on the
batch — which is exactly what the upcoming ``index`` command needs, and
what this module provides.

One pass, six acts per file::

    discover()  ->  load  ->  classify against the registry  ->
    register the row  ->  pipeline.run()  ->  mark INDEXED or FAILED

Three decisions are worth spelling out, because each could reasonably have
gone the other way.

**The row exists before the chunks do.** section 6.1's diagram puts ``create row
status=PENDING`` on the API's side of the enqueue, and the reason is the
crash window: a process that dies mid-write leaves a row stuck
non-terminal, and *that* is what tells the next run to redo the document;
no row at all says nothing. Hence the ordering here — register, walk to
``INDEXING``, *then* run the pipeline — and the matching invariant that a
registry failure aborts this document only: chunks are never written
without a row to own them.

**Classification seeds from ``INDEXED`` rows only.** A row's presence says
"some record exists"; only ``INDEXED`` says "these bytes are in the vector
index". Seeding from every row would make a document whose previous attempt
*failed* classify as ``unchanged`` and never be retried — the record would
keep claiming failure while the file sat unindexed forever. Status lookups
still see every row; only the deduplication view is filtered, and a
retry therefore lands as ``new`` and rewrites cleanly.

**Statuses are recorded in the order they become true.** The lifecycle
graph forbids ``PENDING -> INDEXED``, so the walk passes through ``PARSING``
and ``INDEXING``; because the pipeline is one atomic call from the outside,
both are recorded at the hand-off, and ``INDEXED`` only *after* the write
returns. That ordering — never claiming success before the write — is what
makes a crash recoverable. ``FAILED`` always carries ``code: reason``.

What this module deliberately does *not* do: no file store. A directory
named on a command line is read in place — copying the corpus into
``data/uploads`` would double every byte on disk for no benefit, and
section 6.1's ``FileStore`` belongs to the upload flow on the API side. An emptied
or re-broken *previously indexed* file likewise keeps its old chunks and
its terminal row: removing them is a deletion's job, not a batch pass's,
and the outcome reports the problem either way.

Per-file isolation follows section 5.2.5: one broken file never stops the batch,
and only deliberate :class:`~app.domain.errors.RAGError` is caught — a
defect keeps its traceback. Nothing here touches the network unless the
caller passes an embedder that does.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.domain.errors import RAGError
from app.domain.models import Document, IngestStatus, make_doc_id
from app.embeddings.base import Embedder
from app.ingestion.deduplication import DedupDecision, classify, find_by_doc_id
from app.ingestion.discovery import DiscoveredFile, UnsupportedFile, discover
from app.ingestion.pipeline import (
    RENAMED_CODE,
    UNCHANGED_CODE,
    ChunkSplitter,
    ChunkStore,
    IngestPipeline,
    PipelineAction,
)
from app.ingestion.service import FileResult, ingest_file
from app.observability.logging import get_logger
from app.storage.registry import DocumentRegistry

__all__ = [
    "IndexOutcome",
    "IndexResult",
    "index_directory",
]

logger = get_logger(__name__)

#: Registry listings page at this size; the pass walks every page so a
#: corpus larger than one page is classified against *all* known rows.
_PAGE_SIZE = 500


@dataclass(frozen=True, slots=True)
class IndexOutcome:
    """What happened to one supported file during the pass.

    One outcome per file discovery attempted — a load failure and an empty
    file are outcomes too, so a caller never has to reconstruct the batch
    from absences.

    Parameters
    ----------
    file:
        The load-pass result, carrying provenance (relative path, media
        type, size) and — when one was produced — the document itself.
    doc_id:
        Derived from the file's relative path before any work happened, so
        it is present even for a failure that occurred before
        classification.
    action:
        The pass's verdict in the pipeline's vocabulary: ``created`` /
        ``replaced`` for writes, ``skipped`` for a file that needs nothing,
        ``failed`` for one that could not be indexed. For files the
        pipeline never ran this is derived honestly: an unreadable file is
        ``failed``, an empty one ``skipped``.
    decision:
        The deduplication verdict for files that had text to classify;
        ``None`` when classification never ran (a load failure or an empty
        file have no content to hash).
    chunk_count:
        Chunks written this pass. Zero when nothing reached the store — a
        refusal *before* the write (provider down, registry down) reports
        zero — while a failure that only lost the ``INDEXED``
        confirmation still reports what actually landed, because the
        store, not the status, is what the count describes.
    status:
        The registry row's status *after* the pass; ``None`` when no row
        exists (nothing failed, empty, unchanged-at-a-new-id or renamed
        leaves no row behind).
    code / reason:
        The machine-readable explanation for a skip or failure — the
        pipeline's or service's stable codes, or the typed error's code.
        ``None`` for successful writes, which need no explanation.
    """

    file: FileResult
    doc_id: str
    action: PipelineAction
    decision: DedupDecision | None = None
    chunk_count: int = 0
    status: IngestStatus | None = None
    code: str | None = None
    reason: str | None = None

    @property
    def is_created(self) -> bool:
        """True when this document's chunks were written for the first time."""
        return self.action is PipelineAction.CREATED

    @property
    def is_replaced(self) -> bool:
        """True when an edited document's previous chunks were replaced."""
        return self.action is PipelineAction.REPLACED

    @property
    def is_skipped(self) -> bool:
        """True when this pass wrote nothing, for a documented reason."""
        return self.action is PipelineAction.SKIPPED

    @property
    def is_failed(self) -> bool:
        """True when this file could not be brought into the index."""
        return self.action is PipelineAction.FAILED

    @property
    def written(self) -> bool:
        """True when chunks actually reached the store."""
        return self.is_created or self.is_replaced

    def summary(self) -> dict[str, object]:
        """JSON-safe summary for logs and the index report.

        Ids, the relative path, counts and the outcome — never document
        text and never an absolute path.
        """
        payload: dict[str, object] = {
            "relative_path": self.file.relative_path,
            "doc_id": self.doc_id,
            "action": str(self.action),
            "chunk_count": self.chunk_count,
            "status": str(self.status) if self.status is not None else None,
        }
        if self.decision is not None:
            payload["kind"] = str(self.decision.kind)
        if self.code is not None:
            payload["code"] = self.code
        if self.reason is not None:
            payload["reason"] = self.reason
        return payload


@dataclass(frozen=True, slots=True)
class IndexResult:
    """Everything one pass over a root produced.

    Parameters
    ----------
    root:
        The root's **name only**, never an absolute path — same rule as
        :class:`app.ingestion.service.IngestResult`.
    outcomes:
        One per supported file, in discovery order.
    unsupported:
        Files discovery declined to consider (an extension we do not
        handle), untouched by the pass and never counted as failures.
    """

    root: str
    outcomes: tuple[IndexOutcome, ...]
    unsupported: tuple[UnsupportedFile, ...]

    @property
    def created_count(self) -> int:
        """Documents written for the first time."""
        return sum(1 for outcome in self.outcomes if outcome.is_created)

    @property
    def replaced_count(self) -> int:
        """Documents whose previous chunks were replaced."""
        return sum(1 for outcome in self.outcomes if outcome.is_replaced)

    @property
    def skipped_count(self) -> int:
        """Files that needed no work (unchanged, renamed, empty)."""
        return sum(1 for outcome in self.outcomes if outcome.is_skipped)

    @property
    def failed_count(self) -> int:
        """Files that could not be indexed, for any documented reason."""
        return sum(1 for outcome in self.outcomes if outcome.is_failed)

    @property
    def chunks_written(self) -> int:
        """Chunks that reached the store this pass."""
        return sum(outcome.chunk_count for outcome in self.outcomes)

    @property
    def total_count(self) -> int:
        """Every file discovery reported, attempted or not."""
        return len(self.outcomes) + len(self.unsupported)

    @property
    def has_failures(self) -> bool:
        """True when at least one file could not be indexed.

        The exit-status signal for a command: a pass that indexed 99 files
        and failed one is not a *successful* run, even though it is not a
        total failure either.
        """
        return self.failed_count > 0

    @property
    def decisions(self) -> tuple[DedupDecision, ...]:
        """The deduplication verdicts for files that had text to classify.

        In batch order — what section 5.2.7's report tallies when it is given
        ``decisions``.
        """
        return tuple(outcome.decision for outcome in self.outcomes if outcome.decision is not None)

    def summary(self) -> dict[str, object]:
        """Counts plus one entry per document — never document text."""
        return {
            "root": self.root,
            "total": self.total_count,
            "created": self.created_count,
            "replaced": self.replaced_count,
            "skipped": self.skipped_count,
            "failed": self.failed_count,
            "unsupported": len(self.unsupported),
            "chunks_written": self.chunks_written,
            "documents": [outcome.summary() for outcome in self.outcomes],
        }


def index_directory(
    root: str | Path,
    *,
    embedder: Embedder,
    store: ChunkStore,
    registry: DocumentRegistry,
    chunker: ChunkSplitter | None = None,
    recursive: bool = True,
) -> IndexResult:
    """Discover, classify and index every supported file under ``root``.

    The batch orchestration the storage slice (section 5.3) and section 5.16's pipeline
    were built to serve: discovery decides *which* files, the loader
    decides *what is in them*, deduplication decides *whether anything is
    needed*, and the pipeline does the writing while the registry records
    the document's lifecycle around it.

    Parameters
    ----------
    root:
        Directory to scan.
    embedder:
        section 5.12's contract — one shared instance for the whole batch, so a
        provider sees one client and batching stays coherent.
    store:
        Where chunks land (section 5.3's write path).
    registry:
        Where document rows live; also the source both classification and
        status reporting read from.
    chunker:
        Anything with :meth:`~app.ingestion.pipeline.ChunkSplitter.split`.
        ``None`` means section 5.11's default chunker, whose defaults mirror
        ``Settings`` — the same choice the pipeline makes.
    recursive:
        Whether to descend into subdirectories.

    Returns
    -------
    IndexResult
        One :class:`IndexOutcome` per supported file, plus discovery's
        unsupported list. Always returned for a valid root, even when every
        file failed — an all-failures batch is information, not an
        exception.

    Raises
    ------
    NotFoundError
        If ``root`` does not exist.
    ValidationError
        If ``root`` is a file rather than a directory.
    DependencyError
        If the registry cannot be listed at all — without the known set
        the pass cannot even decide what needs doing, so this is a
        *cannot run* error, not a per-file outcome.

    Notes
    -----
    Per-file problems never raise: they become ``failed`` outcomes with
    stable codes, exactly as section 5.2.5's service does. Caller errors (a bad
    ``root``) and infrastructure errors (a dead registry) propagate — the
    caller is the one who can act on those.
    """
    discovery = discover(root, recursive=recursive)
    records = _all_records(registry)
    # Two views of the same rows, for two different questions: status
    # lookups see *every* row (a stalled one must be reported), while
    # deduplication sees only the indexed ones (a failed one must be
    # retried). See the module docstring.
    known = [record for record in records if record.status is IngestStatus.INDEXED]
    pipeline = IngestPipeline(embedder=embedder, store=store, chunker=chunker, known=known)

    outcomes: list[IndexOutcome] = []
    for found in discovery.files:
        outcomes.append(
            _index_one(found, registry=registry, records=records, known=known, pipeline=pipeline)
        )

    result = IndexResult(
        root=discovery.root.name,
        outcomes=tuple(outcomes),
        unsupported=discovery.unsupported,
    )
    # DEBUG deliberately: a successful pass is the expected case, and a
    # caller piping the report must not find chatter waiting on stderr.
    logger.debug(
        "index pass finished",
        root=result.root,
        created=result.created_count,
        replaced=result.replaced_count,
        skipped=result.skipped_count,
        failed=result.failed_count,
        chunks_written=result.chunks_written,
    )
    return result


def _all_records(registry: DocumentRegistry) -> list[Document]:
    """Every row in the registry, walking the listing's pages.

    ``DocumentRegistry.list`` bounds its result by default (a default page
    is still a bound), and a classification view missing rows because the
    corpus outgrew one page would re-index already-indexed documents.
    """
    records: list[Document] = []
    offset = 0
    while True:
        page = registry.list(limit=_PAGE_SIZE, offset=offset)
        records.extend(page)
        if len(page) < _PAGE_SIZE:
            return records
        offset += len(page)


def _error_line(code: str | None, reason: str | None) -> str:
    """One line for the registry's ``error`` column: ``code: reason``.

    section 6.1's own example shape (``PARSE_ERROR: encrypted pdf``). Always
    non-empty, because the lifecycle refuses a ``FAILED`` transition with
    no message.
    """
    if code and reason:
        return f"{code}: {reason}"
    if code:
        return code
    if reason:
        return reason
    return "indexing failed"


def _retire_stalled_row(
    registry: DocumentRegistry,
    doc_id: str,
    existing: Document | None,
    *,
    error: str,
) -> IngestStatus | None:
    """Move a *non-terminal* row to ``FAILED`` after an unrecoverable file problem.

    A row left ``PENDING``/``PARSING``/``INDEXING`` by a crashed earlier
    run must not stay stalled forever when the file it describes cannot be
    read any more — this is what retires it, so the next run starts from a
    truthful terminal record instead of a promise.

    A row that is already terminal stays exactly as it is: the lifecycle
    deliberately forbids nudging a finished document (section 5.1), and the load
    failure is reported through the outcome regardless. If even this
    transition fails, the row's current status is returned rather than
    masked — the outcome carries the load failure, which is the fact the
    user can act on.
    """
    if existing is None or existing.status.is_terminal:
        return existing.status if existing is not None else None
    try:
        registry.update_status(doc_id, IngestStatus.FAILED, error=error)
    except RAGError:
        return existing.status
    return IngestStatus.FAILED


def _index_one(
    found: DiscoveredFile,
    *,
    registry: DocumentRegistry,
    records: list[Document],
    known: list[Document],
    pipeline: IngestPipeline,
) -> IndexOutcome:
    """Run the six acts over one file and return its outcome.

    ``records`` is the pre-batch snapshot (status lookups); ``known`` is
    the live deduplication view, appended to when this batch writes a
    document so a later file with identical content is recognised as a
    rename within the same pass.
    """
    file_result = ingest_file(
        found.path,
        media_type=found.media_type,
        relative_path=found.relative_path,
        size_bytes=found.size_bytes,
    )
    doc_id = make_doc_id(found.relative_path)
    existing = find_by_doc_id(records, doc_id)

    if file_result.is_failed:
        # The file could not be read, so no document exists to index —
        # but a row left mid-lifecycle by an earlier run is retired here,
        # so a broken file cannot strand its own record.
        status = _retire_stalled_row(
            registry,
            doc_id,
            existing,
            error=_error_line(file_result.code, file_result.reason),
        )
        return IndexOutcome(
            file=file_result,
            doc_id=doc_id,
            action=PipelineAction.FAILED,
            status=status,
            code=file_result.code,
            reason=file_result.reason,
        )

    if file_result.is_skipped:
        # Valid file, no text: nothing became a document, so there is
        # nothing to register or retire. Any pre-existing row keeps its
        # status — the old chunks it describes are still in the store.
        return IndexOutcome(
            file=file_result,
            doc_id=doc_id,
            action=PipelineAction.SKIPPED,
            status=existing.status if existing is not None else None,
            code=file_result.code,
            reason=file_result.reason,
        )

    # Loaded: there is text to classify and, if needed, to register.
    text = file_result.text
    decision = classify(known, source_path=found.relative_path, content=text)

    if not decision.needs_ingest:
        code = UNCHANGED_CODE if decision.is_unchanged else RENAMED_CODE
        return IndexOutcome(
            file=file_result,
            doc_id=doc_id,
            action=PipelineAction.SKIPPED,
            decision=decision,
            status=existing.status if existing is not None else None,
            code=code,
            reason=decision.reason,
        )

    document = Document.create(
        source_path=found.relative_path,
        filename=file_result.source_name,
        media_type=file_result.media_type,
        content=text,
        page_count=file_result.page_count,
    )

    # Register BEFORE writing: this ordering is the crash window's whole
    # story (section 6.1), and a registry that refuses the row stops *this*
    # document before a single chunk can be orphaned without one.
    try:
        registry.create(document)
        registry.update_status(doc_id, IngestStatus.PARSING)
        registry.update_status(doc_id, IngestStatus.INDEXING)
    except RAGError as exc:
        return IndexOutcome(
            file=file_result,
            doc_id=doc_id,
            action=PipelineAction.FAILED,
            decision=decision,
            status=_current_status(registry, doc_id),
            code=exc.code,
            reason=exc.message,
        )

    # ``chunk_count`` is filled in as soon as the pipeline returns: if the
    # *status write* is what fails, the chunks still reached the store and
    # the outcome must not claim otherwise (only ``INDEXED`` is withheld).
    chunk_count = 0
    try:
        outcome = pipeline.run(
            found.path,
            source_path=found.relative_path,
            media_type=found.media_type,
        )
        chunk_count = outcome.chunk_count
        if outcome.is_created or outcome.is_replaced:
            assert outcome.document is not None  # successful writes always carry one
            registry.update_status(doc_id, IngestStatus.INDEXED)
            status = IngestStatus.INDEXED
            # The live view follows the write so a later file holding the
            # same bytes is seen as a rename, not indexed a second time.
            known[:] = [row for row in known if row.doc_id != doc_id]
            known.append(outcome.document)
        else:
            error = _error_line(outcome.code, outcome.reason)
            registry.update_status(doc_id, IngestStatus.FAILED, error=error)
            status = IngestStatus.FAILED
        return IndexOutcome(
            file=file_result,
            doc_id=doc_id,
            action=outcome.action,
            decision=decision,
            chunk_count=chunk_count,
            status=status,
            code=outcome.code,
            reason=outcome.reason,
        )
    except RAGError as exc:
        # The pipeline reports its own failures as results; what can still
        # raise here is the status write itself. The row keeps whichever
        # state it actually reached — never ``INDEXED`` unverified — while
        # ``chunk_count`` still tells the truth about the store.
        return IndexOutcome(
            file=file_result,
            doc_id=doc_id,
            action=PipelineAction.FAILED,
            decision=decision,
            chunk_count=chunk_count,
            status=_current_status(registry, doc_id),
            code=exc.code,
            reason=exc.message,
        )


def _current_status(registry: DocumentRegistry, doc_id: str) -> IngestStatus | None:
    """The row's real status, or ``None`` when it cannot even be read.

    Used on error paths, where the honest answer is whichever state the
    row demonstrably reached rather than the state we hoped it reached —
    and ``None`` when the registry is too broken to say (never ``INDEXED``
    by assumption).
    """
    try:
        record = registry.get(doc_id)
    except RAGError:
        return None
    return record.status if record is not None else None
