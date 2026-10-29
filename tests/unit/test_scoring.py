"""Tests for distance-to-score conversion (§5.4, Stage 32 — scoring).

The conversion is structural: cosine distance has a provable fixed range
[0, 2] for unit-normalised embeddings (the opposite-direction case is
exactly 2; the same-direction case is exactly 0). A linear rescale to
[0, 1] is therefore invertible, preserves order, and never produces
negative scores for positive distances — which is exactly the property
`thresholds.py` (Stage 34) relies on when it applies a floor.
"""

from __future__ import annotations

import pytest

from app.retrieval.scoring import (
    COSINE_DISTANCE_MAX,
    COSINE_DISTANCE_MIN,
    RETRIEVAL_SCORE_RANGE,
    ScoreConverter,
    distance_to_score,
)

# --- Structural properties (the justification) -------------------------------


def test_score_range_is_the_documented_tuple() -> None:
    assert RETRIEVAL_SCORE_RANGE == (0.0, 1.0)


def test_cosine_distance_range_is_fixed() -> None:
    assert COSINE_DISTANCE_MIN == 0.0
    assert COSINE_DISTANCE_MAX == 2.0


# --- Conversion: exact boundary cases ---------------------------------------


def test_distance_zero_maps_to_score_one() -> None:
    """Identical direction (same chunk as query) → maximum score."""
    assert distance_to_score(0.0) == 1.0


def test_distance_two_maps_to_score_zero() -> None:
    """Opposite direction → minimum score."""
    assert distance_to_score(2.0) == 0.0


def test_distance_one_maps_to_score_half() -> None:
    """Mid-range distance (orthogonal vectors) → mid-range score."""
    assert distance_to_score(1.0) == pytest.approx(0.5)


# --- Conversion: monotonicity and linearity ----------------------------------


def test_score_decreases_monotonically_with_distance() -> None:
    """A larger distance must never yield a larger score."""
    converter = ScoreConverter()
    for d1, d2 in [(0.1, 0.5), (0.5, 1.0), (1.2, 1.9)]:
        assert converter.to_score(d1) > converter.to_score(d2)


def test_score_is_linear_in_distance() -> None:
    """Equal steps in distance produce equal steps in score."""
    converter = ScoreConverter()
    delta = 0.5
    scores = [converter.to_score(d) for d in (0.0, delta, delta * 2, delta * 3, delta * 4)]
    diffs = [scores[i + 1] - scores[i] for i in range(len(scores) - 1)]
    # All steps equal (linear rescale), within floating-point tolerance.
    for d in diffs[1:]:
        assert d == pytest.approx(diffs[0], abs=1e-9)


# --- Error policy ------------------------------------------------------------


def test_non_numeric_distance_raises_type_error() -> None:
    with pytest.raises(TypeError) as excinfo:
        distance_to_score("close")  # type: ignore[arg-type]
    assert "float" in str(excinfo.value)


def test_negative_distance_below_tolerance_raises() -> None:
    """A clearly negative distance is impossible for cosine; it must not be
    silently clamped — it indicates a corrupt index or broken embedding.
    """
    with pytest.raises(ValueError) as excinfo:
        distance_to_score(-1.0)
    assert "floor" in str(excinfo.value).lower() or "cosine" in str(excinfo.value).lower()


def test_negative_near_zero_is_tolerated() -> None:
    """Floating-point artifacts near zero (e.g. chromadb returning
    -2e-07 for exact matches) are accepted rather than treated as errors.
    This is the same tolerance `RetrievedChunk.__post_init__` uses.
    """
    result = distance_to_score(-2e-7)
    assert result == pytest.approx(1.0, abs=1e-6)


def test_distance_above_ceiling_raises() -> None:
    with pytest.raises(ValueError) as excinfo:
        distance_to_score(2.1)
    assert "ceiling" in str(excinfo.value).lower() or "exceeds" in str(excinfo.value).lower()


# --- Module-level convenience ------------------------------------------------


def test_module_level_function_exists() -> None:
    assert callable(distance_to_score)
    assert callable(ScoreConverter().to_score)
