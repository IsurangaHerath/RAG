"""Dev-only retrieval inspector — not routed (§5.4, Stage 35 — diagnostics).

Not an API endpoint, not wired into `retriever.py`, not called by any
pipeline stage. A debugging and demonstration tool: given a query, a
`ChromaIndex` and an optional `filter`, it prints (or returns) the
retrieval result as a human-readable breakdown — chunk provenance,
raw cosine distance (§5.3.5), converted score (`scoring.py`, Stage 32),
metadata, and which MMR step selected it (`retriever.py`, Stage 31) —
so a developer can see why a particular chunk won and whether a missing
result is a retrieval problem (bad query embedding, too-restrictive
filter, threshold too high) or a corpus problem (no relevant chunks
indexed).

The output is typed (`RetrievalInspection`) and serializable (JSON-safe),
because it may be written to a sample output file, committed alongside a
corpus, or compared in an evaluation harness — never to a production log,
because `similarity_search` results contain document text (`content`),
which must not reach logs unredacted (§5.9).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from app.retrieval.filters import validate_filter
from app.retrieval.retriever import MMRRetriever, RetrievalQuery
from app.retrieval.scoring import distance_to_score
from app.storage.chroma import ChromaIndex

__all__ = ["RETRIEVAL_INSPECTION_MAX_PREVIEW", "RetrievalInspection", "inspect_retrieval"]

#: How many characters of `content` to include in the preview; never the
#: full text, because inspection output is written to files that may be
#: committed and must not become a second unredacted copy of the corpus.
RETRIEVAL_INSPECTION_MAX_PREVIEW = 200


@dataclass(frozen=True, slots=True)
class ChunkInspection:
    """One retrieved chunk as it appears in an inspection report.

    Carries provenance (ids, file name, page/section when present), the
    raw distance (§5.3.5), the converted score (`scoring.py`, Stage 32),
    a truncated content preview (never full text), and the filter that
    narrowed the result set. Immutable: inspection results are shared
    across threads without mutation risk.
    """

    chunk_id: str
    doc_id: str
    content_preview: str
    distance: float
    score: float
    filename: str | None
    page: int | None
    section: str | None
    filter_applied: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RetrievalInspection:
    """The full inspection result for one query.

    Contains the query text, the filter applied (if any), the number of
    chunks the index holds that matched the filter, the selected chunks
    (with provenance and scores), and the MMR diversity lambda that selected
    them. Immutable and JSON-safe for comparison and commitment.
    """

    query: str
    filter_applied: dict[str, Any] | None
    index_chunk_count: int
    selected_chunks: Sequence[ChunkInspection]
    lambda_diversity: float
    k_requested: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "filter_applied": self.filter_applied,
            "index_chunk_count": self.index_chunk_count,
            "selected_chunks": [c.to_dict() for c in self.selected_chunks],
            "lambda_diversity": self.lambda_diversity,
            "k_requested": self.k_requested,
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


def inspect_retrieval(
    query: str,
    *,
    index: ChromaIndex,
    k: int = 20,
    filter_applied: dict[str, Any] | None = None,
    lambda_diversity: float = 0.7,
    max_preview: int = RETRIEVAL_INSPECTION_MAX_PREVIEW,
) -> RetrievalInspection:
    """Inspect what a query would retrieve — dev/debug only, not routed.

    Builds a `RetrievalQuery`, validates the filter (if any) via the same
    `FilterValidator` that the retrieval layer uses (§5.4, Stage 33),
    runs `MMRRetriever.retrieve`, and packages the result as an
    `RetrievalInspection`. The function never modifies the index, never
    writes to storage, and never calls a live model — it uses the embedder
    held by the index (§5.3.5, Stage 30), which is the offline `FakeEmbedder`
    in tests and the real adapter in production.

    The `content_preview` is truncated (default 200 chars) so inspection
    output — which may be written to a file or committed — never becomes
    a second unredacted copy of the full corpus (§5.9).

    Raises
    ------
    ValidationError
        The filter is malformed (`invalid_filter`, same code as `filters.py`).
    ConfigurationError
        The index has no embedder (`embedder_not_configured`, same as
        `similarity_search`).
    ValueError
        The MMR lambda is outside [0, 1] or `k` is not `>= 1`.
    """
    # Same filter validation as the retrieval layer (§5.4, Stage 33).
    validate_filter(filter_applied)

    # Same query-shape validation as RetrievalQuery (§5.4).
    if not isinstance(query, str) or not query.strip():
        raise ValueError(
            "inspection query must be a non-empty string; "
            "an empty query has no direction to compare against stored chunks"
        )
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError(
            f"inspection k must be an integer >= 1, got {k!r}; "
            f"a non-positive k asks for zero evidence, which is never useful"
        )
    if not (0.0 <= lambda_diversity <= 1.0):
        raise ValueError(
            f"MMR lambda must be in [0, 1], got {lambda_diversity!r}; "
            f"values outside [0, 1] break the relevance-diversity trade-off"
        )

    # Delegate to the concrete retriever (§5.4, Stage 31) — same path the
    # API would take, but without routing through any endpoint.

    retriever = MMRRetriever(index=index, embedder=index._embedder)  # type: ignore[attr-defined, arg-type]
    chunks = retriever.retrieve(
        RetrievalQuery(query=query, filter=filter_applied),
        k=k,
        lambda_diversity=lambda_diversity,
    )

    # Package as inspection: provenance + scores + truncated preview.
    selected = [
        ChunkInspection(
            chunk_id=c.chunk_id,
            doc_id=c.doc_id,
            content_preview=c.content[:max_preview]
            + (" ..." if len(c.content) > max_preview else ""),
            distance=c.distance,
            score=distance_to_score(c.distance),
            filename=c.filename,
            page=c.page,
            section=c.section,
            filter_applied=dict(filter_applied or {}),
        )
        for c in chunks
    ]

    return RetrievalInspection(
        query=query,
        filter_applied=dict(filter_applied or {}),
        index_chunk_count=index.count(),
        selected_chunks=selected,
        lambda_diversity=lambda_diversity,
        k_requested=k,
    )
