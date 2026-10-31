"""Optional score threshold (section 5.4, Stage 34) — `thresholds.py`.

After `scoring.py` (Stage 32) converts distance to a [0, 1] score and
before `reranker` (optional) or `ContextBuilder` (section 5.4), a threshold
excludes evidence that is not close enough to the query. The floor is
optional (`None` = disabled, the default) and configurable per call,
because a calibrated threshold only arrives once evaluation (Stage 45+)
has measured recall at different floors.

The threshold operates on the *converted* score, never on the raw
distance — mixing the two meanings (distance near 0 is good, score near 0
is bad) would break the retrieval chain. A chunk at distance 0.0 has
score 1.0 (maximum); a chunk at distance 2.0 has score 0.0 (minimum);
anything in between is linear. A floor of 0.5 means "only chunks at least
half as close as the nearest possible", which is the intuitive meaning a
caller expects.

Parameters
----------
threshold (float | None):
    The minimum acceptable score; chunks below are excluded. Disabled by
    default (``None``) until evaluation calibrates it (section 5.4, Stage 34).
"""

from __future__ import annotations

from app.domain.models import RetrievedChunk
from app.retrieval.scoring import ScoreConverter

__all__ = ["RETRIEVAL_THRESHOLD_DEFAULT", "apply_threshold"]

#: The default: no threshold applied until evaluation calibrates one.
RETRIEVAL_THRESHOLD_DEFAULT: float | None = None


def apply_threshold(
    chunks: list[RetrievedChunk],
    *,
    threshold: float | None = RETRIEVAL_THRESHOLD_DEFAULT,
) -> list[RetrievedChunk]:
    """Return only chunks whose score meets the floor.

    The conversion uses `scoring.py` (Stage 32) — distance is converted to
    score locally so the same mapping applies everywhere. A missing
    (`None``) threshold returns the full list unchanged; a negative
    threshold is rejected (`ValueError`) because it has no semantic
    meaning; a threshold above 1.0 is also rejected (would exclude
    everything including the exact-match chunk at score 1.0).

    Raises
    ------
    ValueError
        Threshold is outside [0, 1] (no semantic meaning for scores).
    """
    if threshold is None:
        return list(chunks)
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        raise TypeError(f"threshold must be a float or None, got {type(threshold).__name__}")
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError(
            f"threshold must be in [0, 1], got {threshold!r}; "
            f"values above 1.0 exclude everything (even exact matches), "
            f"and negative values have no semantic meaning for scores"
        )
    converter = ScoreConverter()
    return [
        chunk
        for chunk in chunks
        if converter.to_score(chunk.distance) >= threshold - 1e-9  # float tolerance
    ]
