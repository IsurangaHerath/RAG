"""The full query pipeline — retrieve → assemble → prompt → generate (§5.5, Stage 37+).

Not an endpoint (§5.6): this module connects the stages the user has
been building (retrieval: 30-35; generation: 36), producing an
`GroundedContext` and a prompt that the `LLMClient` (Stage 37, `llm_client.py`)
can turn into an answer. The concrete `GroundedAnswerPipeline` runs:

    query → RetrievalQuery (optional filter) → MMRRetriever.retrieve
    → SimpleContextBuilder.build → GroundingPromptBuilder.build
    → LLMClient.generate_messages

No live LLM is called in this file; the concrete pipeline is designed
for offline demonstration, evaluation, and CI — the same constraint
that applies to `retriever.py`, `scoring.py`, and `diagnostics.py`.
A real deployment would inject a live `LLMClient` (built from settings
with `require_openai_key()`); the pipeline stays unchanged.

Error policy: every stage raises typed `RAGError` subclasses (`ValidationError`
for bad input; `ConfigurationError` for missing embedder/LLM; `DependencyError`
for provider failures) so a single `try/except` at the API layer (§5.6)
produces one RFC7807-style response regardless of where the failure
occurred.
"""

from __future__ import annotations

from typing import Any, Protocol

from langchain_core.messages import BaseMessage

from app.domain.errors import ConfigurationError, ValidationError
from app.domain.models import GroundedContext
from app.generation.context_builder import SimpleContextBuilder
from app.generation.llm_client import LLMClient
from app.generation.prompts import GroundingPromptBuilder
from app.retrieval.filters import validate_filter
from app.retrieval.retriever import MMRRetriever, RetrievalQuery
from app.storage.chroma import ChromaIndex

__all__ = ["GroundedAnswerPipeline", "QueryPipeline"]


class QueryPipeline(Protocol):
    """The full §6.2 flow: validate → retrieve → assemble → prompt."""

    def run(
        self,
        query: str,
        *,
        filter_applied: dict[str, Any] | None = None,
        k: int = 20,
        budget_tokens: int = 3000,
    ) -> GroundedContext: ...


class GroundedAnswerPipeline:
    """Concrete pipeline wiring all prior stages (§5.3-§5.5, §6.2).

    Holds the `ChromaIndex` (Stage 30, read + write), the `Embedder`
    (shared with index for same-space embeddings, §5.3.5), the
    `MMRRetriever` (Stage 31), `SimpleContextBuilder` (Stage 36),
    `GroundingPromptBuilder` (§5.5, `prompts.py`), and an optional
    `LLMClient` (Stage 37). All can be injected; defaults build from
    `Settings` when available.

    The pipeline does not call the live LLM by default (`LLMClient` can
    be `None`); when `None`, `run()` stops at the `GroundedContext` stage
    and returns it, which is exactly what evaluation harnesses
    (§5.8) and the CLI demonstration need.
    """

    def __init__(
        self,
        index: ChromaIndex,
        llm_client: LLMClient | None = None,
        embedder: object = None,
    ) -> None:
        if index._embedder is None:
            raise ConfigurationError(
                "Pipeline requires an index with an embedder; construct ChromaIndex(embedder=...)",
                code="embedder_not_configured",
            )
        # The retriever uses the index's embedder (§5.3.5); the pipeline
        # does not create a separate one, so query and chunk embeddings
        # stay in the same space.
        self._retriever = MMRRetriever(index=index, embedder=index._embedder)
        self._builder = SimpleContextBuilder()
        self._prompt_builder = GroundingPromptBuilder()
        self._llm = llm_client

    def retrieve_context(
        self,
        query: str,
        *,
        filter_applied: dict[str, Any] | None = None,
        k: int = 20,
        budget_tokens: int = 3000,
    ) -> GroundedContext:
        """Retrieve and assemble evidence (§6.2); does not call LLM.

        Returns the `GroundedContext` with citation mapping, chunk previews,
        and budget applied. A caller that only needs the evidence (e.g.
        evaluation harness, CLI demonstration) can stop here; a caller
        that needs an answer passes the context to `LLMClient.generate_messages()`
        through the prompt layer (§5.5).
        """
        # Same validation as the retrieval layer (§5.4, Stage 33).
        validate_filter(filter_applied)
        if not isinstance(query, str) or not query.strip():
            raise ValidationError(
                "query must be a non-empty string",
                code="empty_embedding_input",
            )
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise ValidationError(
                f"k must be an integer >= 1, got {k!r}",
                code="invalid_top_k",
            )
        # Step 1: retrieval (§5.4).
        chunks = self._retriever.retrieve(
            RetrievalQuery(query=query, filter=filter_applied),
            k=k,
        )
        # Step 2: context assembly (§5.4 -> §5.5).
        return self._builder.build(query=query, chunks=chunks, budget_tokens=budget_tokens)

    def build_prompt(
        self,
        query: str,
        *,
        filter_applied: dict[str, Any] | None = None,
        k: int = 20,
        budget_tokens: int = 3000,
    ) -> list[BaseMessage]:
        """Full §6.2 flow through `GroundedContext`; does not call LLM.

        Returns the `list[BaseMessage]` that `LLMClient.generate_messages()`
        consumes (§5.5, `llm_client.py`). A caller that wants an actual
        answer must pass these messages to the LLM adapter; this method
        stops at prompt assembly so the pipeline can be tested offline
        without a live model (same constraint as `retriever.py`, `scoring.py`).
        """
        context = self.retrieve_context(
            query, filter_applied=filter_applied, k=k, budget_tokens=budget_tokens
        )
        return self._prompt_builder.build(query=query, ctx=context)
