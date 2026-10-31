"""Persistent vector indexing over Chroma — write path (Stage 24), read path (Stage 30).

section 5.3 puts two independent stores behind interfaces so either side can be
swapped (Chroma → pgvector, SQLite → Postgres) without the rest of the
system noticing. This module is the first of them: the part of
:class:`~app.domain.models.Document`'s journey that *keeps* the vectors
section 5.15 produced, under the ids section 5.11 stamped on them — and, since Stage 30,
the half that gives them back.

Four decisions carry the weight.

**Raw ``chromadb``, not the LangChain wrapper.** section 5.3's contract is
deliberately ours — ``upsert`` / ``delete_document`` / ``count`` /
``similarity_search`` over :class:`~app.domain.models.Chunk` — and the LangChain integration
(``langchain-chroma`` is pinned for later, when generation and retrieval
may want it) would be a second API to translate into a first. Talking to
the client directly keeps one obvious mapping between our types and
chroma's, and every call in this file is four lines or fewer precisely
because nothing is being abstracted twice.

**Vectors arrive on the write path; the query makes its own.** The index
never embeds for a write: chunks come in with their vectors already
aligned (section 5.15's ``embed_chunks``), so the collection is created *without*
asking chroma for an embedding function and no model is ever downloaded by
a write. The read path embeds exactly one thing — the query — with the
:class:`~app.embeddings.base.Embedder` its constructor was given, and hands
chroma that vector rather than the text, so one embedding implementation
serves both halves.

**Results come back as records, not rows.** ``similarity_search`` returns
:class:`~app.domain.models.RetrievedChunk` values; chroma's two-level list
nesting, its ``documents`` / ``metadatas`` / ``distances`` columns and its
index-owned metadata keys all stay inside this file. The distance travels
**raw** — converting it to a score is the retrieval layer's job (section 5.4), so
there is only ever one number for one distance.

**Metadata is copied verbatim, plus three keys the index owns.** The
chunker attaches ``filename``, ``media_type`` and — where the source has
them — ``page`` / ``section`` (section 5.11, see ``Chunk.metadata``), which is
exactly section 5.3's schema; the index adds ``doc_id``, ``chunk_id`` and
``ingested_at``. Those three are written *after* the chunk's own keys so
a mislabelled chunk can never spoof its own identity: ``doc_id`` is what
``delete_document`` filters on, and a metadata row lying about it would
make the overwrite protocol (section 5.16) delete or spare the wrong document.

Persistence and idempotency
---------------------------
One collection, ``techdocs`` (section 5.3), with a cosine HNSW space — cosine
because retrieval cares about direction, not magnitude, and every
embedding model we might swap to produces normalised-ish vectors whose
magnitude carries no meaning. ``persist_directory`` comes from
``Settings.chroma_persist_dir`` so a re-run finds yesterday's index.

``upsert`` keyed by ``chunk_id`` is the store half of Stage 23's
promise: writing the same chunk twice replaces it, so re-ingesting an
unchanged or edited document converges on the same final state rather
than accumulating copies. ``delete_document`` returns how many rows it
removed (section 5.3) — the count the pipeline reports as ``deleted_count``.

Error policy — one class per kind of trouble
--------------------------------------------
Chroma speaks plain exceptions; callers of this module should only ever
see the shared hierarchy:

* a length mismatch between chunks and vectors is caught *before* chroma
  sees it, as ``embedding_count_mismatch`` — chroma would raise a bare
  ``ValueError``, and every existing holder of that code raises
  ``DependencyError``, so the family stays consistent wherever the
  mismatch is first noticed (the same rule section 5.15 followed);
* chroma's ``InvalidArgumentError`` means a vector disagreed with the
  collection's dimension — on a write, a chunk's; on a search, the query
  vector's — raised as ``embedding_dimension_mismatch``, detail
  preserving chroma's own message;
* a caller's mistake reaches the caller as ``ValidationError``: a blank
  ``k`` (``invalid_top_k``), a filter that is not a mapping or that
  chroma rejected (``invalid_filter``, chroma's message kept in
  ``detail``);
* searching on an index built without an ``embedder`` is a
  ``ConfigurationError`` (``embedder_not_configured``) — the write path
  deliberately needs no embedder, so its absence during a search is a
  wiring mistake, not a storage failure;
* anything else from chroma, the filesystem or the underlying sqlite
  store becomes ``DependencyError`` (``dependency_unavailable``), with
  the original message in ``detail`` only.

An empty ``collection_name`` is rejected up front as a
``ValidationError`` rather than whatever chroma might make of it.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, TypeVar

import chromadb
from chromadb.errors import ChromaError, InvalidArgumentError

from app.config import Settings
from app.domain.errors import ConfigurationError, DependencyError, ValidationError
from app.domain.models import Chunk, MetadataFilter, RetrievedChunk
from app.embeddings.base import Embedder

__all__ = ["COLLECTION_NAME", "ChromaIndex"]

#: section 5.3: the single persistent collection every tech-docs chunk lives in.
COLLECTION_NAME: Final = "techdocs"

_T = TypeVar("_T")


class ChromaIndex:
    """Both sides of section 5.3's ``VectorIndex`` over a persistent Chroma store.

    Satisfies the ``ChunkStore`` interface
    :class:`app.ingestion.pipeline.IngestPipeline` declared (section 5.16)
    structurally — same method names, same signatures — so the pipeline's
    store argument moves from the in-memory test double to this class
    without a line of glue. The read half arrived as a *new* method
    (``similarity_search``, section 5.3.5): adding it changed none of these.

    Parameters
    ----------
    persist_directory:
        Where chroma keeps its on-disk index. Defaults to
        ``Settings.chroma_persist_dir`` (``data/chroma`` unless
        configured) — tests always pass an explicit temporary directory
        so no test ever writes the repository's real index.
    collection_name:
        section 5.3 fixes the production name at ``techdocs``; the parameter
        exists so tests and experiments can use isolated collections.
    embedder:
        Used only by :meth:`similarity_search`, to embed the query.
        ``None`` — the default — is the write-only configuration the
        pipeline and the ``index`` command need, since writes arrive with
        vectors already attached; searching such an instance raises
        ``ConfigurationError`` (``embedder_not_configured``) rather than
        quietly skipping the embedding step.

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
        embedder: Embedder | None = None,
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
        self._embedder = embedder
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
        and :meth:`similarity_search` calls ``query`` from here rather
        than growing a second handle to the same data.
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
        what makes re-ingestion converge (section 5.16).

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

    # -- reading ------------------------------------------------------------

    def similarity_search(
        self,
        query: str,
        *,
        k: int,
        filter: MetadataFilter | None = None,
    ) -> list[RetrievedChunk]:
        """Return the stored chunks nearest to ``query``, closest first.

        The read half of section 5.3's contract: embed the query with the
        ``Embedder`` this index was built with, ask chroma for the
        nearest rows (optionally constrained by ``filter``), and hand
        back :class:`~app.domain.models.RetrievedChunk` records.

        Parameters
        ----------
        query:
            The question to search with. Non-blank — the embedder
            enforces that (code ``empty_embedding_input``) before any
            request is made.
        k:
            How many results to return *at most*; an integer ``>= 1``.
            Fewer than ``k`` come back when the index (or the filter)
            holds less: chroma pads nothing and errors on nothing, so an
            over-large ``k`` asks for "everything you have" rather than
            failing.
        filter:
            Metadata constraint handed to chroma as its ``where`` clause —
            ``{"doc_id": {"$in": [...]}, "section": "Requirements"}`` and
            friends (section 5.3). Only its *shape* is checked here: which keys,
            values and operators are legal is section 5.4's filter layer's rule
            to enforce, because that layer is what builds these. An empty
            mapping means no constraint.

        Returns
        -------
        list[RetrievedChunk]
            Nearest first, in chroma's own order, carrying **raw** cosine
            distances — see :class:`~app.domain.models.RetrievedChunk`
            on why converting them to a score is deliberately someone
            else's job.

        Raises
        ------
        ValidationError
            ``k`` is not an integer ``>= 1`` (``invalid_top_k``); the
            filter is not a mapping (``invalid_filter``); or chroma
            rejected the filter's contents (``invalid_filter`` again,
            chroma's message in ``detail``).
        ConfigurationError
            This index was constructed without an embedder
            (``embedder_not_configured``).
        DependencyError
            The query vector's dimension disagrees with the collection's
            (``embedding_dimension_mismatch``), or the store refused the
            read (``dependency_unavailable``).
        """
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValidationError(
                f"k must be an integer >= 1, got {k!r}",
                code="invalid_top_k",
            )
        if filter is not None and not isinstance(filter, Mapping):
            raise ValidationError(
                f"filter must be a mapping, got {type(filter).__name__}",
                code="invalid_filter",
            )
        if self._embedder is None:
            raise ConfigurationError(
                "This index has no embedder, so it cannot embed a query; "
                "construct ChromaIndex(embedder=...) to enable searching.",
                code="embedder_not_configured",
            )

        vector = self._embedder.embed_query(query)
        # An empty mapping means "no constraint", which chroma's where
        # clause has no spelling for; None is it.
        where = dict(filter) if filter else None
        # chroma's stubs type a batch as ``list[Sequence[float]]`` and
        # ``list`` is invariant, so the element is widened explicitly —
        # the same dance ``upsert`` does.
        batch: list[Sequence[float]] = [vector]
        try:
            result = self._run(
                "search",
                lambda: self._collection.query(
                    query_embeddings=batch,
                    n_results=k,
                    where=where,
                    include=["documents", "metadatas", "distances"],
                ),
            )
        except ValueError as exc:
            # Chroma validates where clauses with a bare ValueError
            # ("Expected where operator to be one of ..."), and the only
            # caller-supplied structure in this call is the filter — so
            # this is a caller mistake, not infrastructure trouble.
            raise ValidationError(
                "The vector index rejected the metadata filter.",
                code="invalid_filter",
                detail=str(exc),
            ) from exc
        return self._to_retrieved(result)

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
    def _to_retrieved(result: Any) -> list[RetrievedChunk]:
        """Flatten chroma's nested query result into ordered records.

        chroma returns one inner list per query vector — this path always
        queries with exactly one — and one parallel column per requested
        field, nearest-first. The order is kept rather than re-sorted; an
        empty collection (or a filter that matched nothing) arrives as
        empty columns, which flattens to ``[]`` rather than an error.
        """
        rows = result

        def column(name: str) -> list[Any]:
            nested = rows.get(name)
            return list(nested[0]) if nested else []

        ids = column("ids")
        contents = column("documents")
        metadatas = column("metadatas")
        distances = column("distances")

        # chroma builds the four columns together, so a length mismatch
        # would be a bug in the store rather than bad input — and a bug
        # must look like a bug.
        return [
            RetrievedChunk(
                chunk_id=chunk_id,
                # The id is "<doc_id>:<index>", the same pairing the
                # chunk was written with, so provenance is recovered from
                # the id itself rather than trusted from a second copy in
                # metadata. RetrievedChunk validates the pairing.
                doc_id=chunk_id.partition(":")[0],
                content=content,
                distance=distance,
                metadata=dict(metadata or {}),
            )
            for chunk_id, content, metadata, distance in zip(
                ids, contents, metadatas, distances, strict=True
            )
        ]

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
