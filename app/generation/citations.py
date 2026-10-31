"""Citation parsing and grounding verification for generated answers.

The prompt tells the model to cite every factual claim with ``[n]``
markers that refer to the numbered evidence in ``GroundedContext``
(``prompts.py``). A model can still get that wrong: cite a passage that
was never retrieved, invent a number, or answer with no citations at
all. Those are answer-quality problems, not transport problems, so they
are checked here rather than trusted.

The checker deliberately works on markers, not prose. It proves the
narrow, mechanical part of the grounding contract - every citation
resolves to a chunk that is actually in the context, and a non-abstaining
answer cites something - and leaves claim-level entailment (which would
need another model call) out of scope.

Redaction matters here too: ``CitationCheck`` stores marker numbers and
chunk ids, never answer text, so a failed check can be logged or returned
through the API without copying document content into either.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from app.domain.errors import ValidationError
from app.domain.models import GroundedContext
from app.generation.prompts import ABSTENTION_PHRASE

__all__ = [
    "CitationCheck",
    "GroundingChecker",
    "extract_citation_markers",
    "is_abstention",
    "validate_citations",
]

#: ``[n]`` with one or more digits. Other bracketed text is not a citation.
_CITATION_PATTERN = re.compile(r"\[(\d+)\]")
_WHITESPACE = re.compile(r"\s+")

#: Why a check did not pass. ``None`` means the answer was grounded.
CheckReason = Literal["abstained", "missing_citations", "unknown_citation"]


def _require_text(answer: str) -> str:
    if not isinstance(answer, str):
        raise ValidationError(
            "answer must be a string",
            code="invalid_answer",
        )
    if not answer.strip():
        raise ValidationError(
            "answer must not be empty",
            code="empty_answer",
        )
    return answer


def _normalise(text: str) -> str:
    """Collapse whitespace and case so a faithful abstention is recognised."""
    return _WHITESPACE.sub(" ", text.strip().lower())


def extract_citation_markers(answer: str) -> tuple[int, ...]:
    """Return the distinct ``[n]`` markers in first-appearance order.

    Duplicates are removed because the same passage may be cited several
    times in one answer, and citation validation only cares whether the
    reference resolves. Non-numeric brackets such as ``[note]`` are
    ignored rather than treated as malformed citations - a model writing
    prose about brackets should not fail the check.
    """
    _require_text(answer)
    seen: set[int] = set()
    markers: list[int] = []
    for match in _CITATION_PATTERN.finditer(answer):
        marker = int(match.group(1))
        if marker not in seen:
            seen.add(marker)
            markers.append(marker)
    return tuple(markers)


def is_abstention(answer: str) -> bool:
    """Whether *answer* is the fixed insufficient-information response.

    The comparison ignores surrounding whitespace, repeated spaces and
    case, but not extra sentences: the contract asks for the phrase
    exactly, so an answer that wraps it in invented material is not an
    abstention.
    """
    return _normalise(_require_text(answer)) == _normalise(ABSTENTION_PHRASE)


@dataclass(frozen=True, slots=True)
class CitationCheck:
    """The outcome of validating an answer's citations against its context.

    ``grounded`` is true only when the answer is not an abstention, cites
    at least one passage, and every marker resolves. ``reason`` explains
    a false result in machine-readable form; ``unknown_markers`` lists
    the markers that did not resolve, which is what a caller needs to log
    (the answer text itself is intentionally absent).
    """

    markers: tuple[int, ...]
    cited_chunk_ids: tuple[str, ...]
    unknown_markers: tuple[int, ...]
    abstained: bool
    grounded: bool
    reason: CheckReason | None = None

    def to_dict(self) -> dict[str, object]:
        """JSON-safe summary with no answer text."""
        return {
            "markers": list(self.markers),
            "cited_chunk_ids": list(self.cited_chunk_ids),
            "unknown_markers": list(self.unknown_markers),
            "abstained": self.abstained,
            "grounded": self.grounded,
            "reason": self.reason,
        }


class GroundingChecker:
    """Validate ``[n]`` citations against the retrieved evidence set.

    ``check()`` matches the protocol sketched in the architecture and
    returns a boolean. ``inspect()`` returns the full ``CitationCheck``
    for callers that need to report *why* an answer was rejected.
    """

    def check(self, answer: str, ctx: GroundedContext) -> bool:
        """True only if every citation resolves and the answer is not an abstention."""
        return self.inspect(answer, ctx).grounded

    def inspect(self, answer: str, ctx: GroundedContext) -> CitationCheck:
        """Classify *answer* against *ctx* without raising on bad citations."""
        if is_abstention(answer):
            return CitationCheck(
                markers=(),
                cited_chunk_ids=(),
                unknown_markers=(),
                abstained=True,
                grounded=False,
                reason="abstained",
            )

        markers = extract_citation_markers(answer)
        mapping = ctx.citation_mapping
        unknown = tuple(marker for marker in markers if marker not in mapping)
        cited = tuple(mapping[marker].chunk_id for marker in markers if marker in mapping)

        if unknown:
            return CitationCheck(
                markers=markers,
                cited_chunk_ids=cited,
                unknown_markers=unknown,
                abstained=False,
                grounded=False,
                reason="unknown_citation",
            )
        if not markers:
            return CitationCheck(
                markers=(),
                cited_chunk_ids=(),
                unknown_markers=(),
                abstained=False,
                grounded=False,
                reason="missing_citations",
            )
        return CitationCheck(
            markers=markers,
            cited_chunk_ids=cited,
            unknown_markers=(),
            abstained=False,
            grounded=True,
            reason=None,
        )


def validate_citations(answer: str, ctx: GroundedContext) -> CitationCheck:
    """Return the check, raising when the answer cites a non-existent passage.

    A citation to a chunk that is not in the context is the one failure
    the system must not pass through silently: it would present invented
    evidence as verified. Missing citations and abstentions are returned
    as ``grounded=False`` instead, because they are quality signals the
    caller may choose to surface rather than request failures.
    """
    result = GroundingChecker().inspect(answer, ctx)
    if result.unknown_markers:
        markers = ", ".join(f"[{marker}]" for marker in result.unknown_markers)
        raise ValidationError(
            "answer cites sources that are not in the retrieved context",
            code="invalid_citation",
            detail=f"unknown citation markers: {markers}",
        )
    return result
