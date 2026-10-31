"""Regression gate for evaluation metrics (section 5.8).

The gate compares a metric result against configured floors. Falling
below any floor raises `EvaluationRegressionError` with no answer text
and no chunk content — only the metric name, its value and the threshold.
No fabricated benchmark numbers exist; the thresholds are configured,
not measured.
"""

from __future__ import annotations

import pytest

from app.evaluation.metrics import (
    MIN_RECALL_AT_K,
    EvaluationRegressionError,
    check_regression,
)


def test_regression_raises_for_low_precision() -> None:
    with pytest.raises(EvaluationRegressionError) as excinfo:
        check_regression({"precision_at_k": 0.1})
    assert excinfo.value.metric == "precision_at_k"
    assert excinfo.value.value == 0.1


def test_regression_raises_for_low_recall() -> None:
    with pytest.raises(EvaluationRegressionError) as excinfo:
        check_regression({"recall_at_k": 0.2})
    assert excinfo.value.threshold == MIN_RECALL_AT_K


def test_regression_passes_for_all_above_floors() -> None:
    # No exception: all configured metrics meet their floors.
    check_regression(
        {
            "precision_at_k": 0.8,
            "recall_at_k": 0.9,
            "f1_at_k": 0.85,
            "mrr": 1.0,
            "ndcg_at_k": 0.9,
        }
    )


def test_regression_ignores_missing_metrics() -> None:
    # A partial metric dict is valid; only present keys are checked.
    check_regression({"mrr": 0.6})


def test_regression_message_contains_no_document_text() -> None:
    with pytest.raises(EvaluationRegressionError) as excinfo:
        check_regression({"f1_at_k": 0.1}, query_id="payments")
    msg = str(excinfo.value)
    assert "f1_at_k" in msg
    assert "payments" in msg
    assert "fee" not in msg
    assert "settlement" not in msg
