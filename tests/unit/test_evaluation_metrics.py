"""Tests for the evaluation metrics (Â§5.8, Stages 45â€“52).

Every metric is a pure function: no network, no model, no mutable state.
These tests pin the arithmetic contracts â€” a perfect ranking scores 1.0,
an empty relevant set is handled without NaN, and negative or non-integer
`k` raises a clear `ValueError` rather than a silent arithmetic surprise.
No benchmark results exist; no performance numbers are claimed.
"""

from __future__ import annotations

import pytest

from app.evaluation.metrics import (
    f1_at_k,
    mean_reciprocal_rank,
    normalized_dcg_at_k,
    precision_at_k,
    recall_at_k,
    retrieval_metrics_for_query,
)


def test_perfect_precision_recall_and_f1() -> None:
    assert precision_at_k(["a", "b"], ["a", "b"], k=2) == 1.0
    assert recall_at_k(["a", "b"], ["a", "b"], k=2) == 1.0
    assert f1_at_k(["a", "b"], ["a", "b"], k=2) == 1.0


def test_empty_result_has_zero_precision_and_recall() -> None:
    assert precision_at_k([], ["a"], k=5) == 0.0
    assert recall_at_k([], ["a"], k=5) == 0.0
    assert f1_at_k([], ["a"], k=5) == 0.0


def test_empty_relevant_set_is_perfect_recall() -> None:
    # The design decision from the spec: no relevant chunks means a
    # perfect retrieval (nothing was missed), not NaN.
    assert recall_at_k(["a"], [], k=3) == 1.0
    assert precision_at_k(["a"], [], k=3) == 0.0


def test_recall_is_capped_at_one_for_partial_retrieval() -> None:
    assert recall_at_k(["a"], ["a", "b"], k=5) == 0.5
    assert recall_at_k(["a", "b", "c"], ["a", "b"], k=5) == 1.0


def test_precision_drops_when_irrelevant_items_appear_first() -> None:
    assert precision_at_k(["x", "a"], ["a"], k=2) == 0.5
    assert recall_at_k(["x", "a"], ["a"], k=2) == 1.0


def test_invalid_k_is_rejected() -> None:
    with pytest.raises(ValueError, match="k must be >= 1"):
        precision_at_k(["a"], ["a"], k=0)
    with pytest.raises(ValueError, match="k must be >= 1"):
        precision_at_k(["a"], ["a"], k=-3)


def test_mrr_is_one_for_perfect_first_hit() -> None:
    assert mean_reciprocal_rank(["a"], ["a"]) == 1.0
    assert mean_reciprocal_rank(["a", "b"], ["a"]) == 1.0


def test_mrr_is_zero_when_no_relevant_result() -> None:
    assert mean_reciprocal_rank(["x", "y"], ["a"]) == 0.0


def test_mrr_measures_first_relevant_position() -> None:
    assert mean_reciprocal_rank(["x", "a"], ["a"]) == 0.5
    assert mean_reciprocal_rank(["x", "x", "a"], ["a"]) == pytest.approx(1.0 / 3.0)


def test_empty_relevant_set_has_zero_mrr() -> None:
    assert mean_reciprocal_rank(["a"], []) == 0.0


def test_ndcg_is_one_for_perfect_graded_ranking() -> None:
    # Same grades in same order: ideal ranking equals actual.
    grades = {"a": 2.0, "b": 1.0}
    assert normalized_dcg_at_k(["a", "b"], grades, k=2) == 1.0


def test_ndcg_is_less_than_one_for_reversed_grading() -> None:
    grades = {"a": 2.0, "b": 1.0}
    # Reverse: the high-grade item appears second.
    value = normalized_dcg_at_k(["b", "a"], grades, k=2)
    assert 0.0 < value < 1.0


def test_ndcg_is_zero_for_empty_graded_relevance() -> None:
    assert normalized_dcg_at_k(["a"], {}, k=3) == 0.0


def test_retrieval_metrics_for_query_aggregates_all() -> None:
    result = retrieval_metrics_for_query(
        ["a", "b", "c"],
        ["a", "b"],
        graded_relevance={"a": 2.0, "b": 1.0},
        k=3,
    )
    assert isinstance(result, dict)
    assert "precision_at_k" in result
    assert "mrr" in result
    assert "ndcg_at_k" in result
    assert all(isinstance(value, float) for value in result.values())
    # No answer text is present in any metric payload.
    assert "text" not in result
    assert "chunk" not in result


def test_invalid_ndcg_k_is_rejected() -> None:
    with pytest.raises(ValueError, match="k must be >= 1"):
        normalized_dcg_at_k(["a"], {"a": 1.0}, k=0)
