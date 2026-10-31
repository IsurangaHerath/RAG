"""Retrieval and answer-quality metrics (Â§5.8, Stages 45â€“52).

Every metric is a pure function over a result set: no model call, no
network, no state mutation. That keeps the evaluation harness reproducible
and safe to run in CI without a `OPENAI_API_KEY`. The methods here cover
the family described in ARCHITECTURE.md Â§5.8:

* **Precision@k / Recall@k / F1@k** â€” the standard binary-relevance set
  family; `recall_at_k` is the retrieval contract (Â§5.4) that the index
  must satisfy.
* **MRR (Mean Reciprocal Rank)** â€” rank-aware: the first relevant chunk's
  position is what a user scanning a cited list actually cares about.
* **nDCG (normalized discounted cumulative gain)** â€” graded relevance when
  a chunk is partially relevant; `nDCG@k` divides by the ideal ranking so
  scores are comparable across queries with different numbers of relevant
  chunks.

No performance numbers are reported by this module. Metrics compute
values; the evaluation runner writes them to a CSV/Markdown file. The
README's "not yet measured" statement stays true until a reproducible
golden-set run produces measured numbers that are then committed to the
report.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence

import numpy as np

__all__ = [
    "f1_at_k",
    "mean_reciprocal_rank",
    "normalized_dcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "retrieval_metrics_for_query",
]


def _relevant_ids(expected: Collection[str], k: int | None = None) -> set[str]:
    return {str(item) for item in expected}


#: The binary-relevance family (Â§5.8): precision, recall, F1 at a fixed cut-off.


def precision_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Collection[str],
    k: int = 5,
) -> float:
    """Fraction of the top-k retrieved chunks that are relevant.

    A result is ``float`` in ``[0, 1]``, never ``NaN``. When ``k`` exceeds
    the number of retrieved chunks, the denominator is the number of
    retrieved chunks (not ``k``), because returning nothing relevant is a
    retrieval failure, not an arithmetic artifact.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k!r}")
    top = list(retrieved_ids)[:k]
    if not top:
        return 0.0
    relevant_set = _relevant_ids(relevant_ids)
    return sum(1.0 for item in top if item in relevant_set) / len(top)


def recall_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Collection[str],
    k: int = 5,
) -> float:
    """Fraction of all relevant chunks found in the top-k.

    When no relevant chunks exist, ``recall`` is defined as ``1.0`` (the
    empty set is perfectly recovered) rather than ``NaN``, which avoids
    a missing-value problem in aggregated reports.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k!r}")
    relevant_set = _relevant_ids(relevant_ids)
    if not relevant_set:
        return 1.0
    top = list(retrieved_ids)[:k]
    found = sum(1.0 for item in top if item in relevant_set)
    return found / len(relevant_set)


def f1_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Collection[str],
    k: int = 5,
) -> float:
    """Harmonic mean of ``precision_at_k`` and ``recall_at_k``.

    When both are ``0.0``, ``F1`` is ``0.0`` (the worst case), never ``NaN``.
    """
    p = precision_at_k(retrieved_ids, relevant_ids, k=k)
    r = recall_at_k(retrieved_ids, relevant_ids, k=k)
    if p + r == 0.0:
        return 0.0
    return 2.0 * p * r / (p + r)


#: Rank-aware metrics (Â§5.8).


def mean_reciprocal_rank(
    retrieved_ids: Sequence[str],
    relevant_ids: Collection[str],
) -> float:
    """Average of ``1 / rank`` of the first relevant result.

    A value of ``1.0`` means the very first retrieved chunk is relevant;
    ``0.0`` means no relevant chunk appears (the sum is empty). The
    metric ignores everything after the first hit: a result with a
    relevant chunk at position 1 and nothing else scores perfectly,
    which is the contract a user scanning a citation list expects.
    """
    relevant_set = _relevant_ids(relevant_ids)
    if not relevant_set:
        return 0.0
    for rank, item in enumerate(retrieved_ids, start=1):
        if item in relevant_set:
            return 1.0 / rank
    return 0.0


def normalized_dcg_at_k(
    retrieved_ids: Sequence[str],
    graded_relevance: Mapping[str, float | int],
    k: int = 5,
) -> float:
    """Graded relevance divided by the ideal graded ranking at ``k``.

    ``graded_relevance`` maps a chunk id (or citation label) to its
    relevance grade; grades are non-negative numbers (``0`` = irrelevant,
    positive = relevant with increasing value). The metric compares the
    actual ranking against the same chunks sorted by descending relevance,
    so a perfect ranking yields ``1.0`` regardless of the grade scale.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k!r}")
    grades = [float(graded_relevance.get(str(item), 0.0)) for item in retrieved_ids[:k]]
    actual_dcg = sum(
        grade / float(np.log2(i + 2))  # rank 1 -> log2(3) denominator
        for i, grade in enumerate(grades)
    )
    # Ideal: the same grades sorted descending.
    sorted_grades = sorted(grades, reverse=True)
    ideal_dcg = sum(grade / float(np.log2(i + 2)) for i, grade in enumerate(sorted_grades))
    if ideal_dcg == 0.0:
        return 0.0
    return float(actual_dcg / ideal_dcg)


def retrieval_metrics_for_query(
    retrieved_ids: Sequence[str],
    relevant_ids: Collection[str],
    graded_relevance: Mapping[str, float | int] | None = None,
    k: int = 5,
) -> dict[str, float]:
    """Return all standard retrieval metrics for one query in one call.

    The returned dictionary carries machine-readable float values only â€”
    no answer text, no chunk content â€” matching the redaction contract in
    ``domain/models.py``.
    """
    if graded_relevance is None:
        graded_relevance = {str(item): 1.0 for item in relevant_ids}
    return {
        "precision_at_k": precision_at_k(retrieved_ids, relevant_ids, k=k),
        "recall_at_k": recall_at_k(retrieved_ids, relevant_ids, k=k),
        "f1_at_k": f1_at_k(retrieved_ids, relevant_ids, k=k),
        "mrr": mean_reciprocal_rank(retrieved_ids, relevant_ids),
        "ndcg_at_k": normalized_dcg_at_k(retrieved_ids, graded_relevance, k=k),
        "k": float(k),
        "retrieved_count": float(len(retrieved_ids)),
    }
