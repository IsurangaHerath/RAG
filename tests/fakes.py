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

from app.domain.errors import ValidationError
from app.embeddings.base import require_embedding_text

__all__ = ["FakeEmbedder", "StubEmbeddings"]

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
    * ``mutate`` — corrupts otherwise valid output (wrong count, NaN,
      ragged dimensions) to exercise response validation.

    ``calls`` and ``queries`` record what reached the delegate, so tests
    can assert validation happens *before* the provider would have been
    paid a round trip.
    """

    def __init__(self, *, dimension: int = 64) -> None:
        self._fake = FakeEmbedder(dimension=dimension)
        self.error: BaseException | None = None
        self.mutate: Callable[[list[list[float]]], list[list[float]]] | None = None
        self.calls = 0
        self.queries: list[str] = []

    def embed_documents(
        self, texts: list[str], chunk_size: int | None = None, **kwargs: object
    ) -> list[list[float]]:
        """Return (optionally corrupted) vectors, one per text, in order."""
        self.calls += 1
        if self.error is not None:
            raise self.error
        vectors = self._fake.embed_documents(texts)
        return self.mutate(vectors) if self.mutate else vectors

    def embed_query(self, text: str) -> list[float]:
        """Return (optionally corrupted) single-vector query embedding."""
        self.calls += 1
        self.queries.append(text)
        if self.error is not None:
            raise self.error
        vector = self._fake.embed_query(text)
        if self.mutate:
            return self.mutate([vector])[0]
        return vector
