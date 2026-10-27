"""Idempotent re-ingestion: load → chunk → embed → replace, safely (Stage 23).

Deduplication (§5.2.6) can already *say* what an incoming file is — new,
unchanged, modified or renamed — and the embedding steps (§5.14, §5.15) can
turn chunks into vectors. Nobody yet performs the act those pieces describe:
actually placing a document's chunks into a store, and — the whole point of
this module — placing them a second time without the first copy surviving.

Three rules make that safe.

**Decide before doing.** :func:`classify` runs first, against the documents
this pipeline already knows. An unchanged file returns before the chunker,
the embedder or the store is touched: skipping is what makes re-ingestion
*cheap*, not merely correct.

**Replace, never append.** §6.1's protocol is ``delete_document(doc_id)``
followed by ``upsert`` — old chunk ids and new ones overlap (both count from
``doc_id:0``), so appending would either collide or leave the previous
version's tail retrievable forever. The delete runs even for a
brand-new document: an earlier attempt may have crashed half-written, and a
defensive delete is what makes *retrying* a crash idempotent rather than
duplicating.

**All fallible work first, the destructive step last.** Load, chunk and
embed all happen *before* ``delete_document``. If the provider is down and
retries exhaust, the previous version's chunks are still in the store —
stale beats missing, and the next run simply tries again. What remains is
the narrow window between delete and upsert: a crash there leaves the
document absent until the next run. That is recoverable by running again,
and the planned registry flow closes it for good by marking a document
``INDEXED`` only *after* a successful upsert (§5.2's sketch), so a crash
before that mark is exactly what tells the next run to redo the work.

The known set is an in-memory stand-in for the registry
------------------------------------------------------
:class:`IngestPipeline` keeps the documents it has written in a private
list, seeded from the ``known`` argument. Its lookups reuse
:func:`app.ingestion.deduplication.find_by_doc_id` / ``find_by_hash`` —
named after the registry Protocol in §5.3 on purpose, so Stages 24-27 can
swap a real ``DocumentRegistry`` in place of the list without changing what
this module means. A failed run is *not* remembered, so it is retried; an
empty file is not remembered either, because it never became a document.

**The one rule seeds must follow:** classification hashes the *loaded text*
and :meth:`Document.create` records the same value, so both sides agree by
construction. A caller seeding ``known`` from documents hashed over raw
file *bytes* would make every file read as modified — pass text-derived
hashes, or change both sides together.

**RENAMED is detection, not policy** (§5.2.6). The default here is
``renamed_policy="skip"``: indexing the same bytes under a second path
doubles every retrieval hit. A caller that knows a second path is a
deliberate independent copy can pass ``renamed_policy="ingest"``.

Deviations from §5.2's sketch, stated outright: ``run`` takes a filesystem
path plus an explicit ``source_path`` rather than only ``doc_id`` — there is
no ``FileStore`` yet to resolve ids to files — and its result is named
``DocumentIngestResult`` because service.py already owns ``IngestResult``
for the batch report. Document *status* is left at ``PENDING``: lifecycle
transitions belong to the registry that does not exist yet, which is also
why this module's outcomes are its own vocabulary rather than
:class:`~app.domain.models.IngestStatus`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal, Protocol

from app.domain.errors import RAGError
from app.domain.models import Chunk, Document, make_doc_id
from app.embeddings.base import Embedder
from app.embeddings.generation import embed_chunks
from app.ingestion.deduplication import ChangeKind, classify
from app.ingestion.loaders import TextDocument
from app.ingestion.pdf_loader import PdfDocument
from app.ingestion.service import (
    ingest_file,
)
from app.observability.logging import get_logger
from app.preprocessing.chunking import RecursiveChunker

__all__ = [
    "NO_CHUNKS_CODE",
    "RENAMED_CODE",
    "UNCHANGED_CODE",
    "ChunkSplitter",
    "ChunkStore",
    "DocumentIngestResult",
    "IngestPipeline",
    "PipelineAction",
]

logger = get_logger(__name__)

#: A path already indexed with identical text — nothing to do.
UNCHANGED_CODE: str = "unchanged"
#: The same bytes already live under a different path, and policy says skip.
RENAMED_CODE: str = "renamed"
#: The document loaded but the chunker produced nothing to embed — the file
#: held no splittable text (e.g. only separators). Valid, like an empty file:
#: nothing to index, and not a failure to blame on anyone.
NO_CHUNKS_CODE: str = "no_chunks"


#: What a run did. A deliberately small vocabulary: the *why* lives in
#: ``code``/``reason``, the *what changed* in these four words.
class PipelineAction(StrEnum):
    #: Written for the first time under this ``doc_id``.
    CREATED = "created"
    #: An edited document's previous chunks were deleted, then replaced.
    REPLACED = "replaced"
    #: Nothing was written (unchanged, renamed-skip, empty, no chunks).
    SKIPPED = "skipped"
    #: The run failed; ``code`` carries the typed error's code.
    FAILED = "failed"


class ChunkStore(Protocol):
    """The slice of §5.3's ``VectorIndex`` this stage actually needs.

    Declared consumer-side, here, so ingestion can be tested against an
    in-memory double today; the Chroma adapter (Stages 24+) satisfies it
    structurally with the same method names and signatures, so the swap is
    a matter of construction site, not of contract.
    """

    def upsert(
        self, chunks: Sequence[Chunk], vectors: Sequence[list[float]]
    ) -> None: ...  # pragma: no cover - Protocol declaration, callers supply the body

    def delete_document(
        self, doc_id: str
    ) -> int: ...  # pragma: no cover - Protocol declaration, callers supply the body


class ChunkSplitter(Protocol):
    """§5.2's ``Chunker`` signature, declared where it is consumed."""

    def split(
        self, document: TextDocument | PdfDocument, *, doc_id: str
    ) -> list[Chunk]: ...  # pragma: no cover - Protocol declaration, callers supply the body


@dataclass(frozen=True, slots=True)
class DocumentIngestResult:
    """What one ``run`` did, or refused to do.

    Parameters
    ----------
    source_path:
        The identity path the caller supplied (relative — ``make_doc_id``
        rejects any other kind), echoed because reports point at files.
    doc_id:
        Derived from ``source_path`` before anything else happened, so it
        is present even for a failure that occurred before classification.
    action:
        ``created`` / ``replaced`` / ``skipped`` / ``failed``.
    kind:
        The dedup verdict that drove the run; ``None`` when the run failed
        before classification (the file could not be read at all).
    code:
        Stable machine-readable reason for a skip or failure — the skip
        constants of this module, the service's empty-file codes, or the
        raised error's own code. ``None`` for successful writes.
    reason:
        Human-readable line for the same non-write results. Never contains
        document text; skip reasons come from the decision or the service.
    chunk_count:
        Chunks **written**. Zero for every non-write, including a failure
        that happened after embedding — nothing reached the store.
    deleted_count:
        Chunks the defensive ``delete_document`` removed before writing.
        Non-zero on a "created" run means leftovers from an earlier
        attempt were cleaned up, which is exactly the case it exists for.
    document:
        The record now known to this pipeline after a successful write —
        ``None`` for skips and failures, which is what keeps a failed run
        retryable. Carries full text in memory; :meth:`summary` never
        exposes it.
    """

    source_path: str
    doc_id: str
    action: PipelineAction
    kind: ChangeKind | None = None
    code: str | None = None
    reason: str | None = None
    chunk_count: int = 0
    deleted_count: int = 0
    document: Document | None = None

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
        """True when the run wrote nothing, for any documented reason."""
        return self.action is PipelineAction.SKIPPED

    @property
    def is_failed(self) -> bool:
        """True when the run could not complete."""
        return self.action is PipelineAction.FAILED

    def summary(self) -> dict[str, object]:
        """JSON-safe summary for logs, reports and API responses.

        Ids, paths the caller gave us, counts and the outcome — never
        document text, never an absolute path.
        """
        payload: dict[str, object] = {
            "source_path": self.source_path,
            "doc_id": self.doc_id,
            "action": str(self.action),
            "chunk_count": self.chunk_count,
            "deleted_count": self.deleted_count,
        }
        if self.kind is not None:
            payload["kind"] = str(self.kind)
        if self.code is not None:
            payload["code"] = self.code
        if self.reason is not None:
            payload["reason"] = self.reason
        return payload


class IngestPipeline:
    """One document's journey: decide, build, replace, remember.

    Parameters
    ----------
    embedder:
        §5.12's contract; batching/retries (§5.14) are its own concern.
    store:
        Where chunks land. Only :class:`ChunkStore` is required — the
        §5.3 subset used here.
    chunker:
        Anything with :meth:`ChunkSplitter.split`. Defaults to a fresh
        :class:`~app.preprocessing.chunking.RecursiveChunker`, whose
        defaults mirror ``Settings``.
    known:
        Documents already indexed — the seed that lets ``run`` recognise
        unchanged files on the very first call, and the list a future
        ``DocumentRegistry`` will supply instead.
    renamed_policy:
        ``"skip"`` (default) reports a duplicate path without indexing it;
        ``"ingest"`` writes it independently under its new ``doc_id``.
    """

    def __init__(
        self,
        *,
        embedder: Embedder,
        store: ChunkStore,
        chunker: ChunkSplitter | None = None,
        known: Sequence[Document] = (),
        renamed_policy: Literal["skip", "ingest"] = "skip",
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._chunker: ChunkSplitter = chunker if chunker is not None else RecursiveChunker()
        self._known: list[Document] = list(known)
        if renamed_policy not in ("skip", "ingest"):
            # Not a ValidationError: this is our own constructor being
            # called wrong from code, and Literal already forbids it.
            raise ValueError(f"renamed_policy must be 'skip' or 'ingest', got {renamed_policy!r}")
        self._renamed_policy = renamed_policy

    @property
    def known(self) -> tuple[Document, ...]:
        """Documents this pipeline currently considers indexed.

        What a future registry seeding step inspects, and what the
        "run twice" tests read to prove nothing was recorded twice.
        """
        return tuple(self._known)

    def run(
        self,
        path: str | Path,
        *,
        source_path: str | None = None,
        media_type: str | None = None,
    ) -> DocumentIngestResult:
        """Ingest one file, replacing whatever its ``doc_id`` held.

        Parameters
        ----------
        path:
            Filesystem location to read.
        source_path:
            Identity path — what ``doc_id`` is derived from, in the same
            terms the registry will use (relative, forward slashes).
            Defaults to ``str(path)``, which is only acceptable while the
            path itself is relative; :func:`make_doc_id` rejects absolute
            paths and ``..``, so a machine-specific path cannot silently
            become an identity.
        media_type:
            Override the type derived from the filename, as in
            :func:`app.ingestion.service.ingest_file`.

        Returns
        -------
        DocumentIngestResult
            Always a result; four actions are outcomes, not exceptions.

        Raises
        ------
        ValidationError
            If ``source_path`` (or the fallback) is not a path
            :func:`make_doc_id` accepts — a caller bug, raised before any
            file is opened rather than reported as a per-file failure.

        Notes
        -----
        Only :class:`~app.domain.errors.RAGError` is caught: loaders,
        chunker, embedder and store all speak it. Anything else escaping
        is a defect and is allowed to propagate, per the service's rule.
        """
        identity = source_path if source_path is not None else str(path)
        # Derive the id FIRST: a malformed identity is a caller bug and
        # must raise before any file is opened or any result is minted.
        doc_id = make_doc_id(identity)

        file_result = ingest_file(path, media_type=media_type, relative_path=identity)
        if file_result.is_failed:
            return DocumentIngestResult(
                source_path=identity,
                doc_id=doc_id,
                action=PipelineAction.FAILED,
                code=file_result.code,
                reason=file_result.reason,
            )
        if file_result.is_skipped:
            return DocumentIngestResult(
                source_path=identity,
                doc_id=doc_id,
                action=PipelineAction.SKIPPED,
                code=file_result.code,
                reason=file_result.reason,
            )

        document_in = file_result.document
        assert document_in is not None  # loaded results always carry one
        text = document_in.text
        decision = classify(self._known, source_path=identity, content=text)

        if decision.is_unchanged:
            return DocumentIngestResult(
                source_path=identity,
                doc_id=doc_id,
                action=PipelineAction.SKIPPED,
                kind=decision.kind,
                code=UNCHANGED_CODE,
                reason=decision.reason,
            )
        if decision.is_renamed and self._renamed_policy == "skip":
            return DocumentIngestResult(
                source_path=identity,
                doc_id=doc_id,
                action=PipelineAction.SKIPPED,
                kind=decision.kind,
                code=RENAMED_CODE,
                reason=decision.reason,
            )

        # The write path: every fallible step first, the destructive one
        # last. ``deleted`` starts at zero so a failure before the delete
        # reports honestly that nothing was removed.
        deleted = 0
        try:
            chunks = self._chunker.split(document_in, doc_id=doc_id)
            if not chunks:
                return DocumentIngestResult(
                    source_path=identity,
                    doc_id=doc_id,
                    action=PipelineAction.SKIPPED,
                    kind=decision.kind,
                    code=NO_CHUNKS_CODE,
                    reason="chunker produced no chunks, so there is nothing to embed",
                )
            vectors = embed_chunks(self._embedder, chunks)
            deleted = self._store.delete_document(doc_id)
            self._store.upsert(chunks, vectors)
        except RAGError as exc:
            return DocumentIngestResult(
                source_path=identity,
                doc_id=doc_id,
                action=PipelineAction.FAILED,
                kind=decision.kind,
                code=exc.code,
                reason=exc.message,
                deleted_count=deleted,
            )

        # MODIFIED is the only verdict that replaces: it is the only one
        # whose doc_id was written before. A renamed ingest may re-use the
        # same *text* elsewhere, but under this doc_id it lands for the
        # first time — hence created, not replaced.
        action = PipelineAction.REPLACED if decision.is_modified else PipelineAction.CREATED
        record = Document.create(
            source_path=identity,
            filename=file_result.source_name,
            media_type=file_result.media_type,
            content=text,
            page_count=file_result.page_count,
        )
        self._remember(record)

        logger.info(
            "document ingested",
            doc_id=doc_id,
            action=str(action),
            kind=str(decision.kind),
            chunk_count=len(chunks),
            deleted_count=deleted,
        )
        return DocumentIngestResult(
            source_path=identity,
            doc_id=doc_id,
            action=action,
            kind=decision.kind,
            chunk_count=len(chunks),
            deleted_count=deleted,
            document=record,
        )

    def _remember(self, record: Document) -> None:
        """Add the freshly written record, replacing any earlier one.

        Same ``doc_id`` in, same position in the set out — an edited
        document updates in place rather than accumulating versions,
        which is the in-memory half of "replace, never append".
        """
        self._known = [known for known in self._known if known.doc_id != record.doc_id]
        self._known.append(record)
