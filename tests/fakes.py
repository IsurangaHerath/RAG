"""Deterministic test doubles shared by every test package.

Lives at the ``tests/`` root rather than inside ``tests/unit/`` because
both ``tests/unit`` and the future ``tests/integration`` need it:
integration tests use a real Chroma but must still avoid paying OpenAI
for every run.

None of this ships in ``app/`` — it deliberately understands no meaning,
so it has no business pretending to be a provider in production code.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Callable, Sequence
from types import SimpleNamespace

import openai

from app.domain.errors import ValidationError
from app.domain.models import Chunk
from app.embeddings.base import require_embedding_text

__all__ = [
    "FakeChunkStore",
    "FakeEmbedder",
    "StubEmbeddings",
    "openai_connection_error",
    "openai_status_error",
]

#: Words are the unit of "similarity" for this double. Deliberately naive.
_TOKEN_RE = re.compile(r"\w+")


class FakeEmbedder:
    """An offline embedder that turns shared words into shared directions.

    What it models
    --------------
    Each word is hashed into one of ``dimension`` buckets, buckets are
    counted, and the result is L2-normalised. Two texts that share words
    therefore land at a positive cosine similarity; texts that share none
    land at (or near) zero. That is just enough structure for retrieval
    tests to ask "does the right chunk win?" without a network call.

    What it does not model
    ----------------------
    Any meaning: "car" and "automobile" are unrelated here, because a hash
    knows nothing about words. Golden tests that need real semantics must
    use a real model. Hashing uses SHA-256 rather than Python's ``hash``
    because ``hash`` is salted per process — vectors would differ between
    one run and the next, which is exactly the reproducibility tests need.

    Parameters
    ----------
    dimension:
        Length of every vector this instance returns. Small by default:
        these vectors are only ever compared against each other.
    """

    def __init__(self, *, dimension: int = 64) -> None:
        if dimension < 1:
            raise ValidationError(
                f"embedding dimension must be >= 1, got {dimension}",
                code="invalid_dimension",
            )
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        """Length of every vector this instance returns."""
        return self._dimension

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed each text in order; an empty batch is an empty result."""
        validated = [require_embedding_text(text) for text in texts]
        return [self._vector(text) for text in validated]

    def embed_query(self, text: str) -> list[float]:
        """Embed one query into the same space as document vectors."""
        return self._vector(require_embedding_text(text))

    def _vector(self, text: str) -> list[float]:
        """Hash the words into buckets and normalise to unit length."""
        components = [0.0] * self._dimension
        for token in _TOKEN_RE.findall(text.lower()):
            components[self._bucket(token)] += 1.0
        norm = math.sqrt(sum(component * component for component in components))
        if norm == 0.0:
            # No word characters at all (e.g. "----"): fall back to a fixed
            # direction, because an all-zero vector has no direction and
            # would break every similarity comparison downstream.
            components[0] = 1.0
            return components
        return [component / norm for component in components]

    def _bucket(self, token: str) -> int:
        """Map a word to a bucket index, stably across processes."""
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") % self._dimension


class StubEmbeddings:
    """Stand-in for ``langchain_openai.OpenAIEmbeddings``, the adapter's delegate.

    Shaped like langchain's class — ``embed_documents`` takes ``list[str]``,
    not ``Sequence[str]`` — because that is the protocol
    :class:`~app.embeddings.openai_embedder.OpenAIEmbedder` is typed
    against. Vectors come from :class:`FakeEmbedder`, so contract tests
    exercised through the adapter run on realistic values.

    Knobs cover what a real provider does wrong:

    * ``error`` — raised on every call, to exercise translation of
      provider exceptions into the project's error hierarchy;
    * ``errors`` — a queue: one exception raised per call, then normal
      service resumes, to exercise retry-then-succeed;
    * ``mutate`` — corrupts otherwise valid output (wrong count, NaN,
      ragged dimensions) to exercise response validation.

    ``calls``, ``batches`` and ``queries`` record what reached the
    delegate, so tests can assert validation happens *before* the
    provider would have been paid a round trip, and that batching split
    the work as expected.
    """

    def __init__(self, *, dimension: int = 64) -> None:
        self._fake = FakeEmbedder(dimension=dimension)
        self.error: BaseException | None = None
        self.errors: list[BaseException] = []
        self.mutate: Callable[[list[list[float]]], list[list[float]]] | None = None
        self.calls = 0
        self.batches: list[list[str]] = []
        self.queries: list[str] = []

    def embed_documents(
        self, texts: list[str], chunk_size: int | None = None, **kwargs: object
    ) -> list[list[float]]:
        """Return (optionally corrupted) vectors, one per text, in order."""
        self.calls += 1
        self.batches.append(list(texts))
        if self.errors:
            raise self.errors.pop(0)
        if self.error is not None:
            raise self.error
        vectors = self._fake.embed_documents(texts)
        return self.mutate(vectors) if self.mutate else vectors

    def embed_query(self, text: str) -> list[float]:
        """Return (optionally corrupted) single-vector query embedding."""
        self.calls += 1
        self.queries.append(text)
        if self.errors:
            raise self.errors.pop(0)
        if self.error is not None:
            raise self.error
        vector = self._fake.embed_query(text)
        if self.mutate:
            return self.mutate([vector])[0]
        return vector


def openai_status_error(
    provider_cls: type[openai.APIStatusError],
    status: int,
    *,
    message: str = "provider said no",
) -> openai.APIStatusError:
    """Build an openai HTTP error without a network.

    openai's errors want an ``httpx.Response``; they only read a few
    attributes off it (status code, headers, request), so a plain
    namespace answers just as well and keeps ``httpx`` out of the test
    imports.
    """
    response = SimpleNamespace(
        status_code=status,
        headers={},
        request=SimpleNamespace(url="https://api.openai.com/v1/embeddings", method="POST"),
    )
    return provider_cls(message, response=response, body=None)  # type: ignore[arg-type]


def openai_connection_error(*, message: str = "Connection error.") -> openai.APIConnectionError:
    """Build a connection-failure error (no network, no response object)."""
    return openai.APIConnectionError(
        message=message,
        request=SimpleNamespace(url="https://api.openai.com/v1/embeddings"),  # type: ignore[arg-type]
    )


class FakeChunkStore:
    """In-memory stand-in for the vector index (§5.3's ``VectorIndex``).

    Holds chunks exactly as a store would — ``doc_id`` → ``chunk_id`` →
    vector — and records every operation in order, so tests can assert
    *when* the pipeline's destructive step happens, not merely that it
    did. ``fail_upsert_with`` turns the next writes into a typed failure
    for the failure paths.
    """

    def __init__(self) -> None:
        self.data: dict[str, dict[str, list[float]]] = {}
        self.ops: list[tuple[str, str]] = []
        self.fail_upsert_with: BaseException | None = None

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[list[float]]) -> None:
        if self.fail_upsert_with is not None:
            raise self.fail_upsert_with
        self.ops.append(("upsert", chunks[0].doc_id if chunks else ""))
        for chunk, vector in zip(chunks, vectors, strict=True):
            self.data.setdefault(chunk.doc_id, {})[chunk.chunk_id] = vector

    def delete_document(self, doc_id: str) -> int:
        self.ops.append(("delete", doc_id))
        return len(self.data.pop(doc_id, {}))

    def chunk_ids(self, doc_id: str) -> tuple[str, ...]:
        """Stored chunk ids in insertion order (fresh after a replace)."""
        return tuple(self.data.get(doc_id, {}))

    def count(self, doc_id: str | None = None) -> int:
        """Chunks stored for one document, or across every document."""
        if doc_id is not None:
            return len(self.data.get(doc_id, {}))
        return sum(len(chunks) for chunks in self.data.values())
