"""The embedder contract: rules every implementation must satisfy.

Two kinds of test live here.

**Contract tests** run against *every* implementation registered in
``EMBEDDER_IMPLEMENTATIONS``. A provider added later cannot quietly
disagree with the others about shapes, ordering, or whether queries and
documents share a vector space — the classic bug where retrieval looks
"badly tuned" when the real problem is a mismatched embedder. Adding an
implementation to the registry is one line; ``mypy`` checks it against
the ``Embedder`` Protocol at that same line, and these tests check its
behaviour.

**Double-specific tests** pin the properties later retrieval tests will
lean on from :class:`~tests.fakes.FakeEmbedder`: reproducible vectors
across processes, and shared words meaning shared direction.

No network here — that is the entire point of the double.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest

from app.domain.errors import ValidationError
from app.embeddings.base import Embedder
from tests.fakes import FakeEmbedder

#: Every embedder the tests know how to exercise. A future OpenAI adapter
#: joins by adding one entry — ``mypy`` verifies it here, the contract
#: tests verify its behaviour everywhere else.
EMBEDDER_IMPLEMENTATIONS: dict[str, Callable[[], Embedder]] = {
    "fake": FakeEmbedder,
}

SAMPLE_TEXTS = [
    "Rate limits apply per account.",
    "Deploy with docker compose up -d.",
    "Refunds stay captured for 30 days.",
]


@pytest.fixture(params=sorted(EMBEDDER_IMPLEMENTATIONS))
def embedder(request: pytest.FixtureRequest) -> Embedder:
    """One fresh instance of each registered implementation."""
    factory = EMBEDDER_IMPLEMENTATIONS[request.param]
    return factory()


def _cosine(left: list[float], right: list[float]) -> float:
    """Dot product — both the contract and the double use unit vectors."""
    return sum(a * b for a, b in zip(left, right, strict=True))


# --- Contract: shapes -----------------------------------------------------


def test_one_vector_per_text_in_input_order(embedder: Embedder) -> None:
    """A batch maps 1:1 and in order — chunk *i* must pair with vector *i*."""
    batch = embedder.embed_documents(SAMPLE_TEXTS)
    assert len(batch) == len(SAMPLE_TEXTS)
    for text, vector in zip(SAMPLE_TEXTS, batch, strict=True):
        single = embedder.embed_documents([text])
        assert vector == pytest.approx(single[0])


def test_a_text_embeds_identically_as_query_and_document(embedder: Embedder) -> None:
    """Query space *is* document space, or every similarity score lies."""
    text = SAMPLE_TEXTS[0]
    assert embedder.embed_query(text) == pytest.approx(embedder.embed_documents([text])[0])


def test_vectors_share_one_dimension(embedder: Embedder) -> None:
    """Documents and queries agree on length, or comparison is impossible."""
    vectors = [*embedder.embed_documents(SAMPLE_TEXTS), embedder.embed_query("why?")]
    lengths = {len(vector) for vector in vectors}
    assert len(lengths) == 1
    assert next(iter(lengths)) > 0


def test_components_are_finite_floats(embedder: Embedder) -> None:
    """NaN or inf would poison every later cosine computation silently."""
    vectors = [*embedder.embed_documents(SAMPLE_TEXTS), embedder.embed_query("why?")]
    for vector in vectors:
        for component in vector:
            assert isinstance(component, float)
            assert math.isfinite(component)


def test_no_vector_is_all_zeroes(embedder: Embedder) -> None:
    """A zero vector has no direction; it would rank identically to all."""
    for text in [*SAMPLE_TEXTS, "why?"]:
        vector = embedder.embed_query(text)
        assert any(abs(component) > 0 for component in vector)


def test_distinct_texts_get_distinct_vectors(embedder: Embedder) -> None:
    """Collapsing everything to one vector would make retrieval a no-op."""
    vectors = embedder.embed_documents(SAMPLE_TEXTS)
    assert len({tuple(vector) for vector in vectors}) == len(vectors)


def test_an_empty_batch_is_an_empty_result(embedder: Embedder) -> None:
    """Nothing to embed is not an error condition."""
    assert embedder.embed_documents([]) == []


# --- Contract: input validation -------------------------------------------


def test_blank_batch_element_is_rejected(embedder: Embedder) -> None:
    """Blank input fails locally with a stable code, not at the provider."""
    with pytest.raises(ValidationError) as excinfo:
        embedder.embed_documents(["ok", "   "])
    assert excinfo.value.code == "empty_embedding_input"


def test_blank_query_is_rejected(embedder: Embedder) -> None:
    """An empty question is a caller bug, worth no request."""
    with pytest.raises(ValidationError) as excinfo:
        embedder.embed_query("")
    assert excinfo.value.code == "empty_embedding_input"


def test_non_text_input_is_rejected(embedder: Embedder) -> None:
    """Boundary values arrive untyped from callers; validate anyway."""
    with pytest.raises(ValidationError) as excinfo:
        embedder.embed_documents([None])  # type: ignore[list-item]
    assert excinfo.value.code == "empty_embedding_input"


# --- FakeEmbedder specifics -----------------------------------------------


def test_the_double_is_reproducible_across_instances() -> None:
    """SHA-256, not ``hash``: identical vectors in every run and process."""
    first = FakeEmbedder().embed_documents(SAMPLE_TEXTS)
    second = FakeEmbedder().embed_documents(SAMPLE_TEXTS)
    assert first == second


def test_the_double_gives_shared_words_shared_direction() -> None:
    """The property retrieval tests depend on: overlap beats no overlap."""
    embedder = FakeEmbedder(dimension=128)
    related = _cosine(
        embedder.embed_query("rate limits per account"),
        embedder.embed_query("rate limits apply per account"),
    )
    unrelated = _cosine(
        embedder.embed_query("rate limits per account"),
        embedder.embed_query("docker compose up with systemd"),
    )
    assert related > unrelated


def test_the_double_dimension_is_configurable() -> None:
    """Narrow vectors keep collection tests fast and readable."""
    embedder = FakeEmbedder(dimension=8)
    assert len(embedder.embed_query("anything here")) == 8
    assert embedder.dimension == 8


def test_the_double_normalises_its_vectors() -> None:
    """Unit length means cosine similarity is just the dot product."""
    vector = FakeEmbedder().embed_query("several words in one sentence")
    assert math.sqrt(sum(component * component for component in vector)) == (pytest.approx(1.0))


def test_the_double_survives_text_without_word_characters() -> None:
    """Punctuation-only input still gets a direction, never an all-zero."""
    vector = FakeEmbedder().embed_query("----")
    assert any(abs(component) > 0 for component in vector)


def test_an_impossible_dimension_is_rejected() -> None:
    """Fail loudly at construction rather than return empty vectors."""
    with pytest.raises(ValidationError) as excinfo:
        FakeEmbedder(dimension=0)
    assert excinfo.value.code == "invalid_dimension"
