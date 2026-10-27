"""Tests for the embed step (``app/embeddings/generation.py``).

The module's whole reason to exist is the seam between chunk ids and
vectors, so these tests hammer that seam: order preserved, counts equal,
one shared dimension — and when a deliberately *disobedient* embedder
breaks the contract, the guard fires with the same error codes the
adapter uses. Provider failures must pass through as they arrived.

No network, no API key, no sleeping.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from structlog.testing import capture_logs
from tenacity import wait_none

from app.domain.errors import DependencyError, ProviderRateLimitError
from app.domain.models import Chunk
from app.embeddings.generation import embed_chunks
from app.embeddings.openai_embedder import OpenAIEmbedder
from tests.fakes import FakeEmbedder, StubEmbeddings

DOC_ID = "doc_0123456789abcdef"
MARKER = "CONFIDENTIAL-MARKET-ENTRY"


def make_chunks(count: int) -> list[Chunk]:
    """Build a small document whose text carries a recognisable marker."""
    return [
        Chunk.create(
            doc_id=DOC_ID,
            index=index,
            content=f"chunk {index}: {MARKER} applies per account",
            metadata={"page": index + 1},
        )
        for index in range(count)
    ]


class ScriptedEmbedder:
    """An ``Embedder`` double that records inputs and can break the contract.

    The contract suite keeps *conforming* implementations honest; this
    one is deliberately disobedient so generation's own seam guards can
    be proven to fire, and it records what it was asked so content
    extraction can be proven too.
    """

    def __init__(
        self,
        *,
        vectors: list[list[float]] | None = None,
        error: BaseException | None = None,
    ) -> None:
        self._fake = FakeEmbedder()
        self._vectors = vectors
        self.error = error
        self.batches: list[list[str]] = []

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.batches.append(list(texts))
        if self.error is not None:
            raise self.error
        if self._vectors is not None:
            return self._vectors
        return self._fake.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        if self.error is not None:
            raise self.error
        if self._vectors is not None:
            return self._vectors[0]
        return self._fake.embed_query(text)


# --- The happy path: order preserved, nothing invented --------------------


def test_one_vector_per_chunk_in_order() -> None:
    """What the embedder computes for the contents is exactly what comes back."""
    chunks = make_chunks(6)
    contents = [chunk.content for chunk in chunks]
    assert embed_chunks(FakeEmbedder(), chunks) == FakeEmbedder().embed_documents(contents)


def test_a_chunks_vector_matches_embedding_it_alone() -> None:
    """Position in the batch must not change a chunk's vector."""
    chunks = make_chunks(3)
    vectors = embed_chunks(FakeEmbedder(), chunks)
    alone = FakeEmbedder().embed_documents([chunks[2].content])
    assert vectors[2] == alone[0]


def test_chunk_content_is_what_reaches_the_embedder() -> None:
    """Content is extracted in chunk order — the input to the pairing."""
    chunks = make_chunks(4)
    embedder = ScriptedEmbedder()
    embed_chunks(embedder, chunks)
    assert embedder.batches == [[chunk.content for chunk in chunks]]


def test_empty_chunk_list_short_circuits_without_a_provider_call() -> None:
    """Nothing to embed is an empty result, not a wasted round trip."""
    embedder = ScriptedEmbedder()
    assert embed_chunks(embedder, []) == []
    assert embedder.batches == []


def test_the_returned_vectors_are_the_embedders_untouched() -> None:
    """Generation guards the pairing; it does not reshape anyone's numbers."""
    chunks = make_chunks(3)
    embedder = ScriptedEmbedder()
    assert embed_chunks(embedder, chunks) == embedder._fake.embed_documents(
        [chunk.content for chunk in chunks]
    )


# --- The seam guards: a broken pairing must not ship ----------------------


def test_too_few_vectors_is_a_count_mismatch() -> None:
    """A short result would silently reattach citations to other chunks."""
    chunks = make_chunks(3)
    embedder = ScriptedEmbedder(vectors=[[0.0] * 4, [0.0] * 4])
    with pytest.raises(DependencyError) as excinfo:
        embed_chunks(embedder, chunks)
    error = excinfo.value
    assert error.code == "embedding_count_mismatch"
    assert "2 vector(s) for 3 chunk(s)" in str(error)
    assert error.detail == f"doc_id={DOC_ID}"


def test_too_many_vectors_is_also_a_count_mismatch() -> None:
    """Surplus vectors are the same misalignment, pointed the other way."""
    chunks = make_chunks(2)
    embedder = ScriptedEmbedder(vectors=[[0.0] * 4, [0.0] * 4, [0.0] * 4])
    with pytest.raises(DependencyError) as excinfo:
        embed_chunks(embedder, chunks)
    assert excinfo.value.code == "embedding_count_mismatch"


def test_dimension_drift_across_chunks_is_rejected() -> None:
    """The store would refuse this much later; here the cause is still in view."""
    chunks = make_chunks(3)
    embedder = ScriptedEmbedder(vectors=[[0.0] * 4, [0.0] * 4, [0.0] * 2])
    with pytest.raises(DependencyError) as excinfo:
        embed_chunks(embedder, chunks)
    error = excinfo.value
    assert error.code == "embedding_dimension_mismatch"
    assert "dimension 2 where 4" in str(error)
    assert error.detail == f"chunk_id={DOC_ID}:2"


@pytest.mark.parametrize("position", [0, 1])
def test_an_empty_vector_is_rejected(position: int) -> None:
    """No direction, no cosine — rejected at whichever position it hides."""
    vectors = [[0.0] * 4, [0.0] * 4, [0.0] * 4]
    vectors[position] = []
    chunks = make_chunks(3)
    embedder = ScriptedEmbedder(vectors=vectors)
    with pytest.raises(DependencyError) as excinfo:
        embed_chunks(embedder, chunks)
    error = excinfo.value
    assert error.code == "embedding_invalid_vector"
    assert error.detail == f"chunk_id={DOC_ID}:{position}"


def test_provider_errors_pass_through_unchanged() -> None:
    """A rate limit stays the same instance — one hierarchy, no wrapping."""
    limit = ProviderRateLimitError("still rate limited")
    embedder = ScriptedEmbedder(error=limit)
    with pytest.raises(ProviderRateLimitError) as excinfo:
        embed_chunks(embedder, make_chunks(2))
    assert excinfo.value is limit


# --- Redaction: ids and counts yes, content never -------------------------


@pytest.mark.parametrize(
    "vectors",
    [
        [[0.0] * 4, [0.0] * 4],  # too few for 3 chunks
        [[0.0] * 4, [0.0] * 4, [0.0] * 4, [0.0] * 4],  # too many
        [[0.0] * 4, [0.0] * 4, [0.0] * 2],  # dimension drift
        [[0.0] * 4, [], [0.0] * 4],  # empty vector
    ],
)
def test_failure_text_never_quotes_chunk_content(vectors: list[list[float]]) -> None:
    """Messages and detail reach logs and API responses — ids only, no prose."""
    embedder = ScriptedEmbedder(vectors=vectors)
    with pytest.raises(DependencyError) as excinfo:
        embed_chunks(embedder, make_chunks(3))
    error = excinfo.value
    assert MARKER not in str(error)
    assert MARKER not in (error.detail or "")


def test_log_carries_counts_and_never_content() -> None:
    """One line: which document, how many chunks, what dimension."""
    chunks = make_chunks(4)
    with capture_logs() as captured:
        embed_chunks(FakeEmbedder(), chunks)
    embedded = [entry for entry in captured if entry["event"] == "chunks embedded"]
    assert len(embedded) == 1
    entry = embedded[0]
    assert entry["doc_id"] == DOC_ID
    assert entry["chunk_count"] == 4
    assert entry["dimension"] == FakeEmbedder().dimension
    assert all(MARKER not in str(value) for value in entry.values())


def test_no_log_line_when_there_is_nothing_to_embed() -> None:
    """An empty step is not an event worth paging anyone about."""
    with capture_logs() as captured:
        embed_chunks(FakeEmbedder(), [])
    assert [entry for entry in captured if entry["event"] == "chunks embedded"] == []


# --- End-to-end with the real adapter (still offline) ---------------------


def test_pairs_with_the_adapter_across_batches() -> None:
    """Batches are joined back into one aligned set — the seam's whole job."""
    stub = StubEmbeddings()
    adapter = OpenAIEmbedder(
        stub, model="text-embedding-3-large", batch_size=2, retry_wait=wait_none()
    )
    chunks = make_chunks(5)
    vectors = embed_chunks(adapter, chunks)
    assert stub.batches == [
        [chunks[0].content, chunks[1].content],
        [chunks[2].content, chunks[3].content],
        [chunks[4].content],
    ]
    assert vectors == FakeEmbedder().embed_documents([chunk.content for chunk in chunks])
    assert [len(vector) for vector in vectors] == [FakeEmbedder().dimension] * 5
