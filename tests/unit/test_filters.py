"""Tests for validated metadata filtering (section 5.4, Stage 33 — filters).

The `FilterValidator` enforces the `where` clause shape before it reaches
chroma (`retrieval/retriever.py` forwards unvalidated; this layer validates
before passing). Rules: operators in `VALID_WHERE_OPERATORS`, scalar
operands (str/int/float/bool) or flat scalar lists for `$in`/`$nin`, no
nested mappings or `None` values.
"""

from __future__ import annotations

import pytest

from app.domain.errors import ValidationError
from app.retrieval.filters import VALID_WHERE_OPERATORS, FilterValidator, validate_filter


def test_empty_filter_is_valid() -> None:
    validate_filter(None)
    validate_filter({})


def test_scalar_filter_is_valid() -> None:
    validate_filter({"doc_id": "doc_0123456789abcdef"})
    validate_filter({"section": "Requirements"})


def test_eq_filter_is_valid() -> None:
    validate_filter({"media_type": {"$eq": "text/markdown"}})


def test_in_filter_with_scalar_list_is_valid() -> None:
    validate_filter({"doc_id": {"$in": ["doc_a", "doc_b"]}})


def test_contains_filter_is_valid() -> None:
    validate_filter({"filename": {"$contains": ".md"}})


def test_non_mapping_filter_raises() -> None:
    with pytest.raises(ValidationError) as excinfo:
        validate_filter(["bad"])
    assert excinfo.value.code == "invalid_filter"


def test_invalid_operator_raises() -> None:
    with pytest.raises(ValidationError) as excinfo:
        validate_filter({"doc_id": {"$bogus": 1}})
    assert excinfo.value.code == "invalid_filter"


def test_nested_operand_raises() -> None:
    with pytest.raises(ValidationError) as excinfo:
        validate_filter({"doc_id": {"$in": {"nested": 1}}})
    assert excinfo.value.code == "invalid_filter"


def test_none_operand_raises() -> None:
    with pytest.raises(ValidationError) as excinfo:
        validate_filter({"section": {"$eq": None}})
    assert excinfo.value.code == "invalid_filter"


def test_ambiguous_list_scalar_raises() -> None:
    with pytest.raises(ValidationError) as excinfo:
        validate_filter({"tags": ["a", "b"]})
    assert excinfo.value.code == "invalid_filter"


def test_all_valid_operators_are_accepted() -> None:
    scalar_ops = {"$eq", "$ne", "$gt", "$gte", "$lt", "$lte"}
    list_ops = {"$in", "$nin"}
    string_ops = {"$contains", "$not_contains"}
    for op in VALID_WHERE_OPERATORS:
        if op in scalar_ops:
            FilterValidator().validate({"key": {op: "value"}})
        elif op in list_ops:
            FilterValidator().validate({"key": {op: ["v1", "v2"]}})
        elif op in string_ops:
            FilterValidator().validate({"key": {op: "search"}})
