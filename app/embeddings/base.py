"""The embedder contract: what "turn text into a vector" looks like here.

Why an interface at all
-----------------------
Embedding is the one pipeline step that is pure provider. OpenAI serves it
today, a local model could serve it tomorrow, and nothing else in the
system should notice. Section1 of ARCHITECTURE promises to keep every stage
swappable behind a small interface, and this Protocol is that promise for
embedding: application code asks for an ``Embedder``, never for
``OpenAIEmbeddings``, so changing providers is an import change rather
than a refactor.

Two methods, one space
----------------------
``embed_documents`` turns a batch of chunk texts into one vector each, in
input order; ``embed_query`` turns a single question into one vector. The
split exists so a caller can say what it means — a batch of passages
versus one query — even where the provider uses the same model for both.

The property everything downstream rests on is that **a text embeds the
same way whichever method is called**: ``embed_query(t)`` must equal
``embed_documents([t])[0]``. Retrieval compares query vectors against
document vectors, so two different spaces would make every similarity
score meaningless — the bug would look like bad retrieval, not like a
mismatched embedder.

Shape rules, enforced rather than documented
--------------------------------------------
These hold for every implementation, and
``tests/unit/test_embedder_contract.py`` checks each of them against every
registered embedder:

* one vector per input text, in input order;
* one shared dimensionality across calls and across both methods;
* every component a finite ``float``, and never an all-zero vector (a
  zero vector has no direction to compare);
* an empty batch is an empty result, not an error;
* blank input — or input that is not text at all — raises
  ``ValidationError`` with code ``empty_embedding_input`` before any
  network call, so a caller learns about it for free either way. The
  check itself lives in :func:`require_embedding_text` below, so every
  implementation enforces one rule written once rather than private
  copies that could drift apart.

A new provider joins that suite by adding one line to its registry, which
is how "swappable" stays true instead of aspirational.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from app.domain.errors import ValidationError

__all__ = ["Embedder", "require_embedding_text"]


class Embedder(Protocol):
    """Turn text into vectors of a shared dimensionality.

    Structural (duck-typed): any object with these two methods satisfies
    it, so the real adapter and the offline test double are interchangeable
    without inheriting from anything. ``mypy`` verifies conformance at the
    point of registration in the contract test's registry.
    """

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed every text, returning one vector per input, in order.

        Parameters
        ----------
        texts:
            Passages to embed — chunk text in the indexing path. Must not
            be empty as a *batch*; an empty batch returns ``[]``.

        Raises
        ------
        ValidationError
            Any element is blank or not a string (code
            ``empty_embedding_input``).
        """
        ...  # pragma: no cover - Protocol declaration, callers supply the body

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query into the same space as document vectors.

        Parameters
        ----------
        text:
            One question. Non-blank — an empty question is a caller bug,
            not something to spend a request finding out about.

        Raises
        ------
        ValidationError
            The text is blank or not a string (code
            ``empty_embedding_input``).
        """
        ...  # pragma: no cover - Protocol declaration, callers supply the body


def require_embedding_text(text: str) -> str:
    """Return ``text`` if it is usable, or raise the contract's rejection.

    Shared by every implementation so "blank input fails the same way"
    is one rule in one place. The contract suite would catch three
    private copies drifting apart — but only after a release had already
    shipped three slightly different error messages.

    Raises
    ------
    ValidationError
        ``text`` is blank or not a string; code
        ``empty_embedding_input``.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValidationError(
            "embedding input must be a non-empty string",
            code="empty_embedding_input",
        )
    return text
