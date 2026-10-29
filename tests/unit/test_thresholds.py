"""Tests for optional score threshold (§5.4, Stage 34 — thresholds).

The threshold operates on the converted score (`scoring.py`, Stage 32),
never on raw distance — mixing the two meanings (distance 0 is best,
score 0 is worst) would break retrieval. A `None` threshold is the
calibrated-default-until-evaluation state; a floor in [0, 1] excludes
lower-scoring chunks; out-of-range thresholds raise `ValueError`.
"""

from __future__ import annotations

import pytest

from app.domain.models import RetrievedChunk
from app.retrieval.thresholds import RETRIEVAL_THRESHOLD_DEFAULT, apply_threshold

DOC_ID = "doc_0123456789abcdef"


def make_result(distance: float, content: str = "x") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"{DOC_ID}:0",
        doc_id=DOC_ID,
        content=content,
        distance=distance,
        metadata={"filename": "a.md"},
    )


# --- Default (disabled) ------------------------------------------------------


def test_none_threshold_returns_full_list() -> None:
    chunks = [make_result(0.0), make_result(0.5), make_result(1.5)]
    assert apply_threshold(chunks) == chunks
    assert apply_threshold(chunks, threshold=None) == chunks


# --- Valid thresholds --------------------------------------------------------


def test_threshold_excludes_below_floor() -> None:
    chunks = [make_result(0.0), make_result(0.5), make_result(1.5), make_result(2.0)]
    # Score mapping: 0.0->1.0, 0.5->0.75, 1.5->0.25, 2.0->0.0
    filtered = apply_threshold(chunks, threshold=0.5)
    kept_scores = [1.0 - r.distance / 2.0 for r in filtered]
    assert len(filtered) == 2  # score >= 0.5 means distance <= 1.0
    assert all(s >= 0.5 for s in kept_scores)


def test_exact_floor_keeps_boundary() -> None:
    chunks = [make_result(1.0)]  # distance 1.0 -> score 0.5
    assert len(apply_threshold(chunks, threshold=0.5)) == 1


def test_zero_floor_keeps_everything() -> None:
    chunks = [make_result(2.0)]  # score 0.0
    assert len(apply_threshold(chunks, threshold=0.0)) == 1


def test_one_floor_keeps_only_perfect_matches() -> None:
    chunks = [make_result(0.0), make_result(0.01)]
    assert len(apply_threshold(chunks, threshold=1.0)) == 1
    assert apply_threshold(chunks, threshold=1.0)[0].distance == 0.0


# --- Invalid thresholds ------------------------------------------------------


def test_negative_threshold_raises() -> None:
    with pytest.raises(ValueError) as excinfo:
        apply_threshold([make_result(0.0)], threshold=-0.1)
    assert "[0, 1]" in str(excinfo.value)


def test_above_one_threshold_raises() -> None:
    with pytest.raises(ValueError) as excinfo:
        apply_threshold([make_result(0.0)], threshold=1.5)
    assert "above 1.0" in str(excinfo.value)


def test_non_numeric_threshold_raises() -> None:
    with pytest.raises(TypeError) as excinfo:
        apply_threshold([make_result(0.0)], threshold="high")  # type: ignore[arg-type]
    assert "float" in str(excinfo.value)


def test_default_constant_is_none() -> None:
    assert RETRIEVAL_THRESHOLD_DEFAULT is None
