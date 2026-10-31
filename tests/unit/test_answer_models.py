"""Tests for the answer-side domain models (section 5.1, section 5.5).

`Citation` and `Answer` are the records the API will return, so these
tests pin the properties that make them trustworthy: a citation always
names a real chunk shape, an answer cannot call itself grounded while
citing nothing, and the log-safe summary never carries answer text.
"""

from __future__ import annotations

import json

import pytest

from app.domain.errors import ValidationError
from app.domain.models import Answer, Citation, RetrievedChunk, make_chunk_id

DOC_ID = "doc_0123456789abcdef"


def make_chunk(index: int = 0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=make_chunk_id(DOC_ID, index),
        doc_id=DOC_ID,
        content=f"passage {index}",
        distance=0.1,
        metadata={"filename": f"guide{index}.pdf", "page": index + 1, "section": "Fees"},
    )


def make_citation(marker: int = 1, index: int = 0) -> Citation:
    return Citation.from_chunk(marker, make_chunk(index))


# --- Citation -------------------------------------------------------------


def test_citation_copies_provenance_from_the_chunk() -> None:
    citation = Citation.from_chunk(2, make_chunk(3))

    assert citation.marker == 2
    assert citation.chunk_id == make_chunk(3).chunk_id
    assert citation.doc_id == DOC_ID
    assert citation.filename == "guide3.pdf"
    assert citation.page == 4
    assert citation.section == "Fees"


def test_citation_without_optional_provenance_is_valid() -> None:
    chunk = RetrievedChunk(
        chunk_id=make_chunk_id(DOC_ID, 0),
        doc_id=DOC_ID,
        content="plain text",
        distance=0.0,
    )

    citation = Citation.from_chunk(1, chunk)

    assert citation.filename is None
    assert citation.page is None
    assert citation.section is None


@pytest.mark.parametrize("marker", [0, -1, True, "1", None])
def test_marker_must_be_a_positive_integer(marker: object) -> None:
    with pytest.raises(ValidationError) as excinfo:
        Citation(marker=marker, chunk_id=make_citation().chunk_id, doc_id=DOC_ID)  # type: ignore[arg-type]
    assert excinfo.value.code == "invalid_citation_marker"


def test_citation_refuses_a_chunk_from_another_document() -> None:
    other_doc = "doc_fedcba9876543210"

    with pytest.raises(ValidationError) as excinfo:
        Citation(marker=1, chunk_id=make_chunk_id(other_doc, 0), doc_id=DOC_ID)

    assert excinfo.value.code == "invalid_chunk_id"


def test_citation_refuses_a_malformed_doc_id() -> None:
    with pytest.raises(ValidationError) as excinfo:
        Citation(marker=1, chunk_id="doc_short:0", doc_id="doc_short")

    assert excinfo.value.code == "invalid_doc_id"


def test_citation_refuses_a_padded_chunk_id() -> None:
    with pytest.raises(ValidationError) as excinfo:
        Citation(marker=1, chunk_id=f"{DOC_ID}:007", doc_id=DOC_ID)

    assert excinfo.value.code == "invalid_chunk_id"


@pytest.mark.parametrize("filename", ["", "   "])
def test_citation_refuses_a_blank_filename(filename: str) -> None:
    with pytest.raises(ValidationError) as excinfo:
        Citation(
            marker=1,
            chunk_id=make_citation().chunk_id,
            doc_id=DOC_ID,
            filename=filename,
        )

    assert excinfo.value.code == "invalid_filename"


@pytest.mark.parametrize("page", [0, -2, True])
def test_citation_refuses_an_impossible_page(page: object) -> None:
    with pytest.raises(ValidationError) as excinfo:
        Citation(
            marker=1,
            chunk_id=make_citation().chunk_id,
            doc_id=DOC_ID,
            page=page,  # type: ignore[arg-type]
        )

    assert excinfo.value.code == "invalid_page"


def test_citation_refuses_a_blank_section() -> None:
    with pytest.raises(ValidationError) as excinfo:
        Citation(
            marker=1,
            chunk_id=make_citation().chunk_id,
            doc_id=DOC_ID,
            section=" ",
        )

    assert excinfo.value.code == "invalid_section"


def test_citation_to_dict_is_json_safe_and_has_no_content() -> None:
    payload = make_citation().to_dict()

    assert payload == {
        "marker": 1,
        "chunk_id": make_chunk(0).chunk_id,
        "doc_id": DOC_ID,
        "filename": "guide0.pdf",
        "page": 1,
        "section": "Fees",
    }
    json.dumps(payload)


# --- Answer ---------------------------------------------------------------


def test_grounded_answer_accepts_its_citations() -> None:
    answer = Answer(
        query="what is the fee?",
        text="It is 2% [1].",
        citations=(make_citation(),),
        grounded=True,
    )

    assert answer.grounded is True
    assert answer.citations[0].marker == 1


def test_grounded_without_citations_is_rejected() -> None:
    with pytest.raises(ValidationError) as excinfo:
        Answer(query="q", text="a claim", grounded=True)

    assert excinfo.value.code == "invalid_grounding"


def test_grounded_and_abstained_is_rejected() -> None:
    with pytest.raises(ValidationError) as excinfo:
        Answer(
            query="q",
            text="the fixed phrase",
            citations=(make_citation(),),
            grounded=True,
            abstained=True,
        )

    assert excinfo.value.code == "invalid_grounding"


def test_duplicate_citation_markers_are_rejected() -> None:
    with pytest.raises(ValidationError) as excinfo:
        Answer(
            query="q",
            text="a claim [1][1]",
            citations=(make_citation(1, 0), make_citation(1, 1)),
            grounded=True,
        )

    assert excinfo.value.code == "duplicate_citation_marker"


def test_citations_are_coerced_to_an_immutable_tuple() -> None:
    answer = Answer(
        query="q",
        text="a claim [1]",
        citations=[make_citation()],  # type: ignore[arg-type]
        grounded=True,
    )

    assert isinstance(answer.citations, tuple)


@pytest.mark.parametrize("citations", ["[1]", b"[1]", 42])
def test_citations_must_be_a_sequence_of_records(citations: object) -> None:
    with pytest.raises(ValidationError) as excinfo:
        Answer(query="q", text="t", citations=citations)  # type: ignore[arg-type]

    assert excinfo.value.code == "invalid_citations"


def test_citations_must_contain_citation_records() -> None:
    with pytest.raises(ValidationError) as excinfo:
        Answer(query="q", text="t", citations=("not a citation",))  # type: ignore[arg-type]

    assert excinfo.value.code == "invalid_citations"


@pytest.mark.parametrize(
    ("field", "code"),
    [("query", "empty_query"), ("text", "empty_text")],
)
def test_answer_requires_query_and_text(field: str, code: str) -> None:
    values: dict[str, object] = {"query": "q", "text": "t"}
    values[field] = "   "

    with pytest.raises(ValidationError) as excinfo:
        Answer(**values)  # type: ignore[arg-type]

    assert excinfo.value.code == code


def test_blank_reason_is_rejected() -> None:
    with pytest.raises(ValidationError) as excinfo:
        Answer(query="q", text="t", reason=" ")

    assert excinfo.value.code == "invalid_reason"


def test_answer_to_dict_carries_text_and_citations() -> None:
    answer = Answer(
        query="q",
        text="It is 2% [1].",
        citations=(make_citation(),),
        grounded=True,
    )

    payload = answer.to_dict()

    assert payload["text"] == "It is 2% [1]."
    assert payload["citations"] == [make_citation().to_dict()]
    json.dumps(payload)


def test_answer_summary_never_carries_answer_text() -> None:
    marker = "CONFIDENTIAL-MARKET-ENTRY"
    answer = Answer(
        query="q",
        text=f"The secret is {marker} [1].",
        citations=(make_citation(),),
        grounded=True,
    )

    summary = answer.summary()

    assert marker not in str(summary)
    assert summary["char_count"] == len(answer.text)
    assert summary["grounded"] is True
    json.dumps(summary)


def test_abstained_answer_is_a_valid_ungrounded_record() -> None:
    answer = Answer(
        query="q",
        text="I do not have enough information in the provided context to answer.",
        grounded=False,
        abstained=True,
        reason="abstained",
    )

    assert answer.citations == ()
    assert answer.grounded is False
    assert answer.abstained is True
