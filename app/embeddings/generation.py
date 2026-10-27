"""Chunks to vectors — the pipeline's *embed* step (Stage 22).

§5.2 draws ingestion as ``load → chunk → embed → upsert``. Loading is
§5.2.2/§5.2.3 and splitting is §5.11; this module is the third arrow: it
takes the chunks one document split into and returns one vector per chunk,
in the same order, ready to hand straight to
``VectorIndex.upsert(chunks, vectors)``.

Why a seam and not a comprehension
----------------------------------
``embedder.embed_documents([chunk.content for chunk in chunks])`` really
is one line — and one silent way to poison an index. This is where
*chunk ids* meet *vectors*: if the counts disagree, citation *i* points at
chunk *j*'s text and nothing downstream can notice. The adapter already
validates everything it receives (§5.13), but two gaps remain:

* its checks are per request, while alignment is a property of the whole
  document — batches are joined *here*, so this is the only place that
  sees all of them at once;
* the contract suite (§5.12) proves implementations behave at test time,
  not at 3 a.m. in production, so the worst failure deserves a cheap
  runtime guard of its own.

Only the *cheap and catastrophic* checks live here: counts and each
vector's length — O(chunks), never a walk of every component. The
expensive per-component walk (finiteness) stays where it was already paid
for, in the adapter.

Provider failures pass through untouched: a rate limit that exhausts the
retry policy (§5.14) arrives as ``ProviderRateLimitError`` and leaves as
the same exception, so callers handle one error hierarchy regardless of
which layer noticed first.

The single log line carries counts only — document id, chunk count,
dimension. Chunk *content* never reaches this module's logs, with or
without ``LOG_DOCUMENT_CONTENT``.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.domain.errors import DependencyError
from app.domain.models import Chunk
from app.embeddings.base import Embedder
from app.observability.logging import get_logger

__all__ = ["embed_chunks"]

logger = get_logger(__name__)


def embed_chunks(embedder: Embedder, chunks: Sequence[Chunk]) -> list[list[float]]:
    """Embed every chunk's content, returning one vector per chunk, in order.

    Parameters
    ----------
    embedder:
        Any implementation of §5.12's contract. Batching and retries
        (§5.14) are that implementation's own concern and invisible here.
    chunks:
        The split document — typically one document's worth, as §6.1's
        worker does. Order in is order out: ``vectors[i]`` belongs to
        ``chunks[i]``.

    Returns
    -------
    list[list[float]]
        Exactly what ``VectorIndex.upsert(chunks, vectors)`` pairs back
        with ``chunks``. Empty input is an empty result — no provider
        call, no error, matching the adapter's own short-circuit.

    Raises
    ------
    DependencyError
        The embedder broke the contract: wrong number of vectors
        (``embedding_count_mismatch``), an empty vector
        (``embedding_invalid_vector``), or vectors that disagree on
        dimension (``embedding_dimension_mismatch``). The codes match
        the adapter's (§5.13) on purpose — one alert rule should catch
        the same class of failure wherever it is first noticed.
    RAGErrors from the embedder
        Propagate unchanged (a rate limit stays a rate limit), so the
        caller sees a single error hierarchy.
    """
    if not chunks:
        return []
    vectors = embedder.embed_documents([chunk.content for chunk in chunks])
    _require_pairing(chunks, vectors)
    logger.info(
        "chunks embedded",
        doc_id=chunks[0].doc_id,
        chunk_count=len(chunks),
        dimension=len(vectors[0]),
    )
    return vectors


def _require_pairing(chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
    """Guard the seam: one vector each, all the same non-zero length.

    Both failure modes are silent ones — a count mismatch misattributes
    citations, a mixed dimension is rejected much later by the store,
    far from the cause — so both are answered here, where the inputs are
    still in view. Messages quote ids, positions and counts, never chunk
    content.
    """
    if len(vectors) != len(chunks):
        raise DependencyError(
            f"Embedder returned {len(vectors)} vector(s) for {len(chunks)} chunk(s).",
            code="embedding_count_mismatch",
            detail=f"doc_id={chunks[0].doc_id}",
        )
    dimension = len(vectors[0])
    if dimension == 0:
        raise DependencyError(
            "Embedder returned an empty vector.",
            code="embedding_invalid_vector",
            detail=f"chunk_id={chunks[0].chunk_id}",
        )
    for position, vector in enumerate(vectors[1:], start=1):
        if not vector:
            raise DependencyError(
                "Embedder returned an empty vector.",
                code="embedding_invalid_vector",
                detail=f"chunk_id={chunks[position].chunk_id}",
            )
        if len(vector) != dimension:
            raise DependencyError(
                f"Embedder returned dimension {len(vector)} where {dimension} was expected.",
                code="embedding_dimension_mismatch",
                detail=f"chunk_id={chunks[position].chunk_id}",
            )
