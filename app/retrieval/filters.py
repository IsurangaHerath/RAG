"""Validated metadata filtering (§5.4, Stage 33) — `filters.py`.

The `retriever.py` read path (Stage 31) forwards a `MetadataFilter` to
`ChromaIndex.similarity_search` without validating its contents — that is
deliberate separation: the index knows nothing about which operators or
value shapes are legal (that is a retrieval-layer policy, not a storage
property), and `similarity_search` translates any malformed `where` clause
into a `ValidationError` (`invalid_filter`) with chroma's message
preserved (§5.3.5).

This file enforces the legal shape *before* it reaches chroma, so callers
get a typed, actionable error (`invalid_filter`) rather than relying on
chroma's internal validation message. The allowed operators and value
types match chroma's ``where`` clause contract empirically verified
against the installed version (§5.3.5's error-policy notes):

    Operators: $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin,
               $contains, $not_contains

    Values: string, number (int/float), bool, or a list of any of those.

Nested structures (e.g. ``{"$in": {"nested": 1}}``) and ``None`` values
are rejected — both break chroma's ``where`` parser and must never reach
it. A filter that passes validation is handed straight to
`similarity_search` (no second copy); one that fails is rejected here,
with the specific reason recorded in `detail`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from app.domain.errors import ValidationError
from app.domain.models import MetadataFilter

__all__ = ["VALID_WHERE_OPERATORS", "FilterValidator", "validate_filter"]

#: The ``where`` clause operators chroma's installed version accepts
#: (verified empirically; see §5.3.5 error-policy notes).
VALID_WHERE_OPERATORS: frozenset[str] = frozenset(
    {
        "$eq",
        "$ne",
        "$gt",
        "$gte",
        "$lt",
        "$lte",
        "$in",
        "$nin",
        "$contains",
        "$not_contains",
    }
)


class FilterValidator:
    """Validate a `MetadataFilter` before it reaches chroma's ``where``.

    The validation rules mirror chroma's `where` parser behavior (verified
    empirically against the installed `chromadb` version): operators must
    be from the documented set, operand values must be scalar (str/int/float/
    bool) or a flat list of scalars, and keys must be non-empty strings.

    The validator never modifies the filter — it only confirms legality,
    so the same mapping can be passed through to
    `similarity_search` unchanged. A failure raises `ValidationError`
    (`invalid_filter`) with the chroma-like message preserved in `detail`,
    so a caller that catches `invalid_filter` sees both a typed code and
    the specific operator/value that failed.

    Methods
    -------
    validate(filter) -> None
        Confirms shape; raises `ValidationError` on any violation.
    """

    def validate(self, filter: MetadataFilter | None) -> None:
        """Confirm the filter is a legal `MetadataFilter`; raise otherwise."""
        if filter is None:
            return  # None means "no constraint"; chroma treats it identically.
        if not isinstance(filter, Mapping):
            raise ValidationError(
                f"filter must be a mapping, got {type(filter).__name__}",
                code="invalid_filter",
                detail="chroma expects dict for `where`; got non-mapping filter",
            )
        for key, value in filter.items():
            if not isinstance(key, str) or not key.strip():
                raise ValidationError(
                    f"filter keys must be non-empty strings, got {key!r}",
                    code="invalid_filter",
                    detail=f"chroma requires non-empty string keys in `where`; got {key!r}",
                )
            self._validate_value(key, value)

    def _validate_value(self, key: str, value: Any) -> None:
        """Validate one filter operand value against chroma's rules."""
        if isinstance(value, Mapping):
            # Operator mapping: value is a dict of operator -> operand.
            for op, operand in value.items():
                if op not in VALID_WHERE_OPERATORS:
                    raise ValidationError(
                        f"filter operator '{op}' is not valid",
                        code="invalid_filter",
                        detail=f"Expected one of {sorted(VALID_WHERE_OPERATORS)}; got '{op}'",
                    )
                # Some operators ($contains, $not_contains) expect string
                # operands; $in / $nin expect a list; others expect scalars.
                # The installed chroma validates this internally; our job
                # is to reject visibly broken operand structures before
                # chroma sees them (nested dicts, None, non-scalar where a
                # scalar is expected by every operator).
                if isinstance(operand, dict):
                    raise ValidationError(
                        f"filter operand for '{key}.{op}' must not be nested",
                        code="invalid_filter",
                        detail="Nested structures in `where` operands are not supported by chroma",
                    )
                if operand is None:
                    raise ValidationError(
                        f"filter operand for '{key}.{op}' is None",
                        code="invalid_filter",
                        detail="`None` is not a valid operand value in chroma's `where` clause",
                    )
                # $in / $nin require a list operand; all others accept scalars.
                if op in ("$in", "$nin"):
                    if not isinstance(operand, Sequence) or isinstance(operand, (str, bytes)):
                        raise ValidationError(
                            f"filter operator '{op}' requires a list operand for '{key}'",
                            code="invalid_filter",
                            detail=f"Expected a list for $in/$nin; got {type(operand).__name__}",
                        )
                    # Every element of the list must be a scalar.
                    for idx, element in enumerate(operand):
                        is_nested = isinstance(element, Mapping)
                        is_list = isinstance(element, Sequence) and not isinstance(
                            element, (str, bytes)
                        )
                        if is_nested or is_list:
                            raise ValidationError(
                                f"filter element at index {idx} for '{key}.{op}' must be a scalar",
                                code="invalid_filter",
                                detail="nested/non-scalar list elements not supported",
                            )
                else:
                    # Non-$in/$nin operators: operand must be a scalar.
                    if isinstance(operand, Sequence) and not isinstance(operand, (str, bytes)):
                        raise ValidationError(
                            f"filter operator '{op}' for '{key}' requires a scalar operand",
                            code="invalid_filter",
                            detail=f"Expected str/int/float/bool; got {type(operand).__name__}",
                        )
        else:
            # Scalar value (not an operator mapping) — used by top-level
            # filters that reference a single value directly. The installed
            # chroma treats scalar values differently depending on context,
            # but for this layer the contract is: scalar must be a JSON
            # scalar (str/int/float/bool), never nested, never None.
            if isinstance(value, Mapping):
                raise ValidationError(
                    f"filter value for '{key}' must not be a nested mapping",
                    code="invalid_filter",
                    detail="Nested mappings as scalar filter values break chroma's `where` parser",
                )
            if value is None:
                raise ValidationError(
                    f"filter value for '{key}' is None",
                    code="invalid_filter",
                    detail="`None` is not a valid scalar operand in chroma's `where` clause",
                )
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                # A scalar value that is a list is ambiguous: is it meant
                # for $in or is it a mistaken scalar? The installed chroma
                # treats it differently per operator; here we reject it as
                # ambiguous to force the caller to be explicit (use $in).
                raise ValidationError(
                    f"filter value for '{key}' is a list but no $in/$nin operator is specified; "
                    f"either use $in/$nin or provide a scalar value",
                    code="invalid_filter",
                    detail=f"Ambiguous list operand for '{key}': use $in/$nin or a scalar",
                )


def validate_filter(filter: MetadataFilter | None) -> None:
    """Module-level convenience: validate a filter or pass silently.

    ``None`` means no constraint and is always valid; anything else is
    validated as a `MetadataFilter` mapping (§5.3). This function is the
    entry point the retrieval layer calls before passing a filter to
    `similarity_search` (Stage 30), so an invalid filter fails here with a
    typed `ValidationError` (`invalid_filter`) rather than reaching
    chroma and returning an ambiguous `ValueError` or `InvalidArgumentError`.
    """
    FilterValidator().validate(filter)
