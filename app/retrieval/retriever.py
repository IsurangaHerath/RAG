"""Top-k retrieval with MMR diversity — Stage 31.

The `Retriever` protocol (section 5.4) turns a `RetrievalQuery` (the output of the
optional `QueryAnalyzer`) into the *smallest sufficient* evidence set.
Three steps, all inside this file:

1. **Embed.** The query is embedded with the same `Embedder` the index
   holds (section 5.3.5) so document and query vectors live in the same space —
   the property `test_embedder_contract` enforces.
2. **Fetch.** `ChromaIndex.similarity_search` (Stage 30) gives the nearest
   chunks, with an optional `filter` (section 5.4's query-time constraint, not
   validated here — `filters.py`, Stage 33 enforces operator/value rules).
3. **MMR.** The nearest results tend to cluster around the same passage;
   Maximal Marginal Relevance (section 5.4, Stage 31) trades a small amount of
   relevance for diversity so the evidence set covers more of the corpus.

MMR uses cosine similarity (1 - cosine distance) both for relevance
(query-to-chunk) and redundancy (chunk-to-chunk). The embedder embedded
both the query and every selected chunk, so similarity is computed in the
same space as retrieval — not approximated from the raw distance, which
would mix two meanings of "similar". `lambda = 0.7` keeps 70 % relevance,
30 % diversity, the standard balance for evidence retrieval.

Parameters
----------
k (default 20):
    How many chunks to return. Fewer may come back when the index (or
    filter) holds less; the contract stays "at most k", never exactly k.
lambda (default 0.7):
    The MMR relevance-diversity balance. `1.0` = pure relevance (no MMR);
    `0.0` = pure diversity (no relevance); both extremes are allowed so a
    caller can disable diversity by setting `lambda=1.0`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from app.domain.errors import ConfigurationError, ValidationError
from app.domain.models import MetadataFilter, RetrievedChunk
from app.embeddings.base import Embedder
from app.storage.chroma import ChromaIndex

__all__ = ["MMR_LAMBDA", "RetrievalQuery", "Retriever"]

#: Default MMR relevance-diversity balance (section 5.4).
MMR_LAMBDA = 0.7


@dataclass(frozen=True, slots=True)
class RetrievalQuery:
    """The input to :class:`Retriever` — a standalone question plus a filter.

    The optional `filter` (a `MetadataFilter`, the alias from the domain)
    is applied at query time against the index (section 5.4), not validated here —
    `filters.py` (Stage 33) enforces which operators and value types are
    legal; this layer only passes it through.

    Parameters
    ----------
    query:
        The standalone question to embed and search with. Non-empty — the
        embedder enforces that (``empty_embedding_input``) before any
        request reaches chroma.
    filter:
        Optional metadata constraint for `ChromaIndex.similarity_search`.
        `None` means no constraint; an empty mapping `{}` also means no
        constraint, which chroma treats identically.
    """

    query: str
    filter: MetadataFilter | None = None

    def __post_init__(self) -> None:
        # Frozen, so assign through object.__setattr__ (the only sanctioned
        # mutation path for frozen dataclasses during construction).
        if not isinstance(self.query, str) or not self.query.strip():
            raise ValidationError(
                "retrieval query must be a non-empty string",
                code="empty_embedding_input",
            )
        if self.filter is not None and not isinstance(self.filter, Mapping):
            raise ValidationError(
                f"retrieval filter must be a mapping, got {type(self.filter).__name__}",
                code="invalid_filter",
            )


class Retriever(Protocol):
    """Fetch the smallest sufficient evidence set (section 5.4).

    A concrete retriever (below) holds the `ChromaIndex` and `Embedder`
    needed to run the three-step path; the Protocol only declares the
    public shape, so callers depend on the contract, not the implementation.
    """

    def retrieve(
        self,
        rq: RetrievalQuery,
        *,
        k: int = 20,
        lambda_diversity: float = MMR_LAMBDA,
    ) -> list[RetrievedChunk]: ...  # pragma: no cover


class MMRRetriever:
    """Concrete retriever: embed → chroma top-k → MMR diversity (section 5.4).

    The ``Embedder`` is held at construction time (the same instance the
    index writes through, so query and document embeddings share a space);
    ``similarity_search`` is called on the ``ChromaIndex`` given at
    construction, with the query's embedded vector and the optional
    ``filter`` forwarded unvalidated (the index validates it as a
    ``Mapping``; ``filters.py``, Stage 33 enforces the operator/value
    rules).

    Parameters
    ----------
    index:
        The vector store to search (section 5.3.5). Must have been constructed
        with an ``embedder`` (``ConfigurationError`` ``embedder_not_configured``
        if missing — a wiring mistake, not a storage failure).
    embedder:
        The embedder shared with the index (section 5.3.5's read path). Used only
        to embed the query text into the same space as the stored chunks.
    """

    def __init__(
        self,
        index: ChromaIndex,
        embedder: Embedder,
    ) -> None:
        if index._embedder is None:
            raise ConfigurationError(
                "The index has no embedder; construct it with embedder=...",
                code="embedder_not_configured",
            )
        self._index = index
        self._embedder = embedder

    def retrieve(
        self,
        rq: RetrievalQuery,
        *,
        k: int = 20,
        lambda_diversity: float = MMR_LAMBDA,
    ) -> list[RetrievedChunk]:
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise ValidationError(
                f"retrieval k must be an integer >= 1, got {k!r}",
                code="invalid_top_k",
            )
        if not (0.0 <= lambda_diversity <= 1.0):
            raise ValidationError(
                f"MMR lambda must be in [0, 1], got {lambda_diversity!r}",
                code="invalid_request",
            )
        # Step 1: fetch nearest from storage (section 5.3.5).
        nearest = self._index.similarity_search(
            rq.query,
            k=max(k, 1),
            filter=rq.filter,
        )
        # Step 2: MMR diversity selection (section 5.4).
        selected = self._mmr_select(nearest, k=k, lambda_diversity=lambda_diversity)
        return selected

    def _mmr_select(
        self,
        candidates: list[RetrievedChunk],
        k: int,
        lambda_diversity: float,
    ) -> list[RetrievedChunk]:
        """Select ``k`` chunks with maximal marginal relevance (section 5.4).

        Relevance: cosine similarity to the query (1 - distance). The
        distance travels raw from the index (section 5.3.5); converting it to
        a score is the scoring layer's job (section 5.4, `scoring.py`, Stage 32),
        so this layer converts it locally for the MMR formula and only
        here, never writing a second number back.

        Redundancy: cosine similarity between chunk embeddings, computed
        with the same embedder that wrote them — never approximated from
        the raw distance, which would mix two meanings of "similar".
        """
        selected: list[RetrievedChunk] = []
        remaining = list(candidates)
        while len(selected) < k and remaining:
            best_index = 0
            best_score = -float("inf")
            # Embed the best-so-far selection once per round to avoid
            # re-embedding the same chunk repeatedly.
            selected_embeddings: list[list[float]] = []
            for chunk in selected:
                selected_embeddings.append(self._embedder.embed_documents([chunk.content])[0])
            for idx, chunk in enumerate(remaining):
                # Relevance: larger similarity (closer distance) = better.
                relevance = 1.0 - chunk.distance
                # Redundancy: max similarity to any already-selected chunk.
                redundancy = 0.0
                if selected_embeddings:
                    chunk_embedding = self._embedder.embed_documents([chunk.content])[0]
                    similarities = [
                        self._cosine_similarity(chunk_embedding, sel_emb)
                        for sel_emb in selected_embeddings
                    ]
                    redundancy = max(similarities)
                mmr_score = (lambda_diversity * relevance) - ((1.0 - lambda_diversity) * redundancy)
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_index = idx
            selected.append(remaining.pop(best_index))
        return selected

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Cosine similarity of two vectors (1 = same direction, 0 = orthogonal)."""
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)
