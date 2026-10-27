"""Persistent vector indexing over Chroma — the write path (Stage 24).

§5.3 puts two independent stores behind interfaces so either side can be
swapped (Chroma → pgvector, SQLite → Postgres) without the rest of the
system noticing. This module is the first of them: the part of
:class:`~app.domain.models.Document`'s journey that *keeps* the vectors
§5.15 produced, under the ids §5.11 stamped on them.

Three decisions carry the weight.

**Raw ``chromadb``, not the LangChain wrapper.** §5.3's contract is
deliberately ours — ``upsert`` / ``delete_document`` / ``count`` over
:class:`~app.domain.models.Chunk` — and the LangChain integration
(``langchain-chroma`` is pinned for later, when generation and retrieval
may want it) would be a second API to translate into a first. Talking to
the client directly keeps one obvious mapping between our types and
chroma's, and every call in this file is four lines or fewer precisely
because nothing is being abstracted twice.

**Vectors arrive, never get made here.** The index never embeds: chunks
come in with their vectors already aligned (§5.15's ``embed_chunks``), so
the collection is created *without* asking chroma for an embedding
function and no model is ever downloaded by a write. The same rule will
govern the query path when ``similarity_search`` lands with retrieval:
embed with our :class:`~app.embeddings.base.Embedder`, then hand the
vector over.

**Metadata is copied verbatim, plus three keys the index owns.** The
chunker attaches ``filename``, ``media_type`` and — where the source has
them — ``page`` / ``section`` (§5.11, see ``Chunk.metadata``), which is
exactly §5.3's schema; the index adds ``doc_id``, ``chunk_id`` and
``ingested_at``. Those three are written *after* the chunk's own keys so
a mislabelled chunk can never spoof its own identity: ``doc_id`` is what
``delete_document`` filters on, and a metadata row lying about it would
make the overwrite protocol (§5.16) delete or spare the wrong document.

Persistence and idempotency
---------------------------
One collection, ``techdocs`` (§5.3), with a cosine HNSW space — cosine
because retrieval cares about direction, not magnitude, and every
embedding model we might swap to produces normalised-ish vectors whose
magnitude carries no meaning. ``persist_directory`` comes from
``Settings.chroma_persist_dir`` so a re-run finds yesterday's index.

``upsert`` keyed by ``chunk_id`` is the store half of Stage 23's
promise: writing the same chunk twice replaces it, so re-ingesting an
unchanged or edited document converges on the same final state rather
than accumulating copies. ``delete_document`` returns how many rows it
removed (§5.3) — the count the pipeline reports as ``deleted_count``.

Error policy — one class per kind of trouble
--------------------------------------------
Chroma speaks plain exceptions; callers of this module should only ever
see the shared hierarchy:

* a length mismatch between chunks and vectors is caught *before* chroma
  sees it, as ``embedding_count_mismatch`` — chroma would raise a bare
  ``ValueError``, and every existing holder of that code raises
  ``DependencyError``, so the family stays consistent wherever the
  mismatch is first noticed (the same rule §5.15 followed);
* chroma's ``InvalidArgumentError`` means a vector's dimension disagrees
  with the collection's — raised as ``embedding_dimension_mismatch``,
  detail preserving chroma's own message;
* anything else from chroma, the filesystem or the underlying sqlite
  store becomes ``DependencyError`` (``dependency_unavailable``), with
  the original message in ``detail`` only.

An empty ``collection_name`` is rejected up front as a
``ValidationError`` rather than whatever chroma might make of it.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, TypeVar

import chromadb
from chromadb.errors import ChromaError, InvalidArgumentError

from app.config import Settings
from app.domain.errors import DependencyError, ValidationError
from app.domain.models import Chunk

__all__ = ["COLLECTION_NAME", "ChromaIndex"]

#: §5.3: the single persistent collection every tech-docs chunk lives in.
COLLECTION_NAME: Final = "techdocs"

_T = TypeVar("_T")


class ChromaIndex:
    """Write side of §5.3's ``VectorIndex`` over a persistent Chroma store.

    Satisfies the ``ChunkStore`` interface
    :class:`app.ingestion.pipeline.IngestPipeline` declared (§5.16)
    structurally — same method names, same signatures — so the pipeline's
    store argument moves from the in-memory test double to this class
    without a line of glue. ``similarity_search`` is deliberately absent
    until ``RetrievedChunk`` exists to return (§5.4); adding it is a new
    method, not a change to these.

    Parameters
    ----------
    persist_directory:
        Where chroma keeps its on-disk index. Defaults to
        ``Settings.chroma_persist_dir`` (``data/chroma`` unless
        configured) — tests always pass an explicit temporary directory
        so no test ever writes the repository's real index.
    collection_name:
        §5.3 fixes the production name at ``techdocs``; the parameter
        exists so tests and experiments can use isolated collections.

    Raises
    ------
    ValidationError
        ``collection_name`` is blank.
    DependencyError
        The client or collection could not be opened — a path that is a
        file, an unreadable directory, a corrupt store.
    """

    def __init__(
        self,
        *,
        persist_directory: Path | str | None = None,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        if not collection_name.strip():
            raise ValidationError(
                "collection name must not be empty",
                code="invalid_request",
            )
        directory = (
            Path(persist_directory)
            if persist_directory is not None
            else Settings().chroma_persist_dir
        )
        self._persist_directory = directory
        self._collection = self._run(
            "open",
            lambda: chromadb.PersistentClient(path=str(directory)).get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            ),
        )

    @property
    def collection(self) -> Any:
        """The raw chroma collection.

        An escape hatch on purpose: tests inspect stored rows through it,
        and the query path (§5.4, coming with retrieval) will call
        ``query`` from here rather than growing a second handle to the
        same data.
        """
        return self._collection

    @property
    def persist_directory(self) -> Path:
        """Where this instance's data lives on disk."""
        return self._persist_directory

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[list[float]]) -> None:
        """Write chunks and their vectors, replacing any previous version.

        One row per chunk: id = ``chunk_id``, text = ``content``, the
        metadata described in the module docstring. Nothing is appended —
        a re-write of an existing ``chunk_id`` overwrites it, which is
        what makes re-ingestion converge (§5.16).

        Raises
        ------
        DependencyError
            Counts disagree (``embedding_count_mismatch`` — caught here
            so chroma's bare ``ValueError`` never escapes), or the store
            refused the write (``embedding_dimension_mismatch`` for a
            wrong-sized vector, ``dependency_unavailable`` otherwise).
        """
        if not chunks:
            return
        if len(chunks) != len(vectors):
            raise DependencyError(
                f"Got {len(chunks)} chunk(s) but {len(vectors)} vector(s); "
                "they must be paired one to one before writing.",
                code="embedding_count_mismatch",
                detail=f"doc_id={chunks[0].doc_id}",
            )
        ingested_at = datetime.now(UTC).isoformat()
        # chroma's stub types a batch as ``list[Sequence[float]]`` and
        # ``list`` is invariant, so the elements are re-typed here rather
        # than passing ``list[list[float]]`` through as-is.
        batch: list[Sequence[float]] = list(vectors)
        self._run(
            "write",
            lambda: self._collection.upsert(
                ids=[chunk.chunk_id for chunk in chunks],
                embeddings=batch,
                documents=[chunk.content for chunk in chunks],
                metadatas=[self._item_metadata(chunk, ingested_at) for chunk in chunks],
            ),
        )

    def delete_document(self, doc_id: str) -> int:
        """Remove every chunk stored under ``doc_id``; return how many.

        Zero for a document that is not in the index — deleting nothing
        is a normal part of re-ingestion (the defensive delete of a
        brand-new document), not an error.
        """
        result = self._run(
            "delete",
            lambda: self._collection.delete(where={"doc_id": doc_id}),
        )
        return int(result.get("deleted", 0))

    def count(self) -> int:
        """Total chunks in the collection, across every document."""
        return int(self._run("count", lambda: self._collection.count()))

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _item_metadata(chunk: Chunk, ingested_at: str) -> dict[str, str | int | float | bool]:
        """Chunk metadata plus the index's own three keys, ours last.

        Written last so ``doc_id`` / ``chunk_id`` / ``ingested_at`` cannot
        be shadowed by a chunk that carried keys of the same name —
        ``doc_id`` in particular is the filter ``delete_document`` trusts.
        """
        return {
            **chunk.metadata,
            "doc_id": chunk.doc_id,
            "chunk_id": chunk.chunk_id,
            "ingested_at": ingested_at,
        }

    @staticmethod
    def _run(operation: str, action: Callable[[], _T]) -> _T:
        """Run one chroma call, translating its failures into our errors.

        Chroma's own exceptions (and anything below it — the filesystem,
        sqlite) are infrastructure trouble to a caller, so they surface
        as ``DependencyError``; its ``InvalidArgumentError`` specifically
        means a vector disagreed with the collection's dimension, which
        is reported under the embedder family's code so one alert rule
        still catches mismatched dimensions wherever they appear.
        """
        try:
            return action()
        except InvalidArgumentError as exc:
            raise DependencyError(
                "A vector's dimension does not match the dimension already "
                "stored in the collection; the embedding model in force "
                "appears to have changed.",
                code="embedding_dimension_mismatch",
                detail=str(exc),
            ) from exc
        except (ChromaError, OSError, sqlite3.Error) as exc:
            raise DependencyError(
                f"The vector index could not complete the {operation} operation.",
                detail=f"{type(exc).__name__}: {exc}",
            ) from exc
