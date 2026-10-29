"""Distance to score conversion (§5.4, Stage 32) — justified.

`similarity_search` (Stage 30) returns `RetrievedChunk.distance` as a
**raw cosine distance**: 0.0 when the stored vector points exactly the
same way as the query, 2.0 when it points the opposite way, and anything
between for the angles in between. That number travels unconverted through
the retrieval chain (§5.3.5's docstring, `retriever.py` line 17) — the
retriever's MMR diversity uses cosine *similarity* (1 - distance) locally,
but never writes a second number back; conversion to a single normalized
score is this file's one job.

The mapping is linear and bounded:

    score = 1 - (distance / 2)

which is exactly ``(2 - distance) / 2``. The justification is structural,
not empirical — cosine distance over unit-normalised embeddings has a
fixed, provable range [0, 2] (the extreme opposite-direction case), so a
linear rescale to [0, 1] preserves order, is invertible, and never
surprises a caller with a negative score. Changing the conversion would
mean two different meanings of "similar" in circulation; keeping it here,
as a single public function, is the safeguard.

Parameters
----------
threshold (optional):
    The lowest acceptable score; chunks below it are excluded by the
    retrieval layer (`thresholds.py`, Stage 34). Disabled by default
    (``None``) until a calibrated threshold arrives.
"""

from __future__ import annotations

__all__ = ["RETRIEVAL_SCORE_RANGE", "ScoreConverter", "distance_to_score"]

#: The fixed range of cosine distance for unit-normalised embeddings.
COSINE_DISTANCE_MIN: float = 0.0
COSINE_DISTANCE_MAX: float = 2.0

#: The resulting score range after linear rescaling.
RETRIEVAL_SCORE_RANGE: tuple[float, float] = (0.0, 1.0)


class ScoreConverter:
    """Convert a raw cosine distance (`RetrievedChunk.distance`) to a score.

    The conversion is a single linear rescale against the fixed range
    [0, 2] (§5.4 notes the justification: cosine distance has a provable
    maximum of 2 for opposite-direction unit vectors). It is deliberately
    not configurable — changing the mapping would break the contract that
    one distance has one score, and any calibration should happen at the
    threshold layer (`thresholds.py`, Stage 34), not here.

    Methods
    -------
    to_score(distance: float) -> float:
        ``1.0`` at distance ``0`` (identical direction), ``0.0`` at
        distance ``2`` (opposite direction), linear in between.
        Raises ``ValueError`` for out-of-range distances so a corrupt
        index (e.g. a bug producing a negative distance beyond the
        floating-point tolerance) surfaces as an error rather than a
        silent out-of-range score.
    """

    def to_score(self, distance: float) -> float:
        if not isinstance(distance, (int, float)) or isinstance(distance, bool):
            raise TypeError(f"distance must be a float, got {type(distance).__name__}")
        if distance < COSINE_DISTANCE_MIN - 1e-5:  # near-zero-negative float tolerance
            raise ValueError(
                f"distance {distance!r} is below the cosine-distance floor ({COSINE_DISTANCE_MIN})"
            )
        if distance > COSINE_DISTANCE_MAX + 1e-5:
            raise ValueError(
                f"distance {distance!r} exceeds the cosine-distance ceiling ({COSINE_DISTANCE_MAX})"
            )
        # Linear rescale to [0, 1]: opposite directions become 0, same
        # direction becomes 1. Justified by the fixed [0, 2] range (§5.4).
        score = 1.0 - (distance / COSINE_DISTANCE_MAX)
        return max(0.0, min(1.0, score))


def distance_to_score(distance: float) -> float:
    """Module-level convenience: same conversion, no instance needed."""
    return ScoreConverter().to_score(distance)
