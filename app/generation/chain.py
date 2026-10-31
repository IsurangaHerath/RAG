"""The full query pipeline: retrieve -> assemble -> prompt -> generate (§5.5, §6.2).

Not an endpoint (§5.6): this module connects retrieval (Stages 30-35),
context assembly (Stage 36), prompt construction, the `LLMClient` adapter
and citation verification into one object. The concrete
`GroundedAnswerPipeline` runs:

    query -> RetrievalQuery (optional filter) -> MMRRetriever.retrieve
          -> SimpleContextBuilder.build -> GroundingPromptBuilder.build
          -> LLMClient.generate_messages -> validate_citations
          -> Answer

The pipeline can stop at any leg deliberately: `retrieve_context()` returns
the evidence, `build_prompt()` returns the messages, and `answer()` performs
the model call and verifies its citations. That split keeps evaluation
harnesses and CLI demonstrations offline: they can use the first two
without a live model.

Error policy: every stage raises typed `RAGError` subclasses
(`ValidationError` for bad input; `ConfigurationError` for a missing
embedder/LLM; `DependencyError` for provider failures) so a single
`try/except` at the API layer (§5.6) produces one RFC7807-style response
regardless of where the failure occurred. An answer that cites a passage
outside the retrieved context raises `ValidationError` (`invalid_citation`)
rather than being returned as if it were verified.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any, Protocol

from langchain_core.messages import BaseMessage

from app.domain.errors import ConfigurationError, DependencyError, ValidationError
from app.domain.models import Answer, Citation, GroundedContext
from app.generation.citations import validate_citations
from app.generation.context_builder import SimpleContextBuilder
from app.generation.llm_client import LLMClientProtocol
from app.generation.prompts import GroundingPromptBuilder
from app.retrieval.filters import validate_filter
from app.retrieval.retriever import MMRRetriever, RetrievalQuery
from app.storage.chroma import ChromaIndex

__all__ = ["GroundedAnswerPipeline", "QueryPipeline"]


def _response_text(messages: Sequence[BaseMessage]) -> str:
    """Join the text of the assistant's response messages.

    LangChain messages carry either a string or a list of content blocks
    (text, tool calls, images). Only text blocks become answer text; a
    response with none at all is provider trouble, not an empty answer,
    because returning ``""`` would be indistinguishable from a model that
    deliberately said nothing.
    """
    parts: list[str] = []
    for message in messages:
        content = message.content
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict):
                    value = block.get("text")
                    if isinstance(value, str):
                        parts.append(value)
    text = "\n".join(part.strip() for part in parts if part.strip())
    if not text:
        raise DependencyError(
            "The language model returned no answer text",
            code="empty_llm_response",
        )
    return text


class QueryPipeline(Protocol):
    """The full §6.2 flow: retrieve -> assemble -> prompt -> answer."""

    def retrieve_context(
        self,
        query: str,
        *,
        filter_applied: dict[str, Any] | None = None,
        k: int = 20,
        budget_tokens: int = 3000,
    ) -> GroundedContext: ...

    def build_prompt(
        self,
        query: str,
        *,
        filter_applied: dict[str, Any] | None = None,
        k: int = 20,
        budget_tokens: int = 3000,
    ) -> list[BaseMessage]: ...

    def answer(
        self,
        query: str,
        *,
        filter_applied: dict[str, Any] | None = None,
        k: int = 20,
        budget_tokens: int = 3000,
        history: Sequence[str] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> Answer: ...

    def stream(
        self,
        query: str,
        *,
        filter_applied: dict[str, Any] | None = None,
        k: int = 20,
        budget_tokens: int = 3000,
        history: Sequence[str] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> Iterator[str]: ...


class GroundedAnswerPipeline:
    """Concrete pipeline wiring retrieval, prompting, generation and checking.

    Holds the `ChromaIndex` (Stage 30, read + write), the `Embedder`
    (shared with the index for same-space embeddings, §5.3.5), the
    `MMRRetriever` (Stage 31), `SimpleContextBuilder` (Stage 36),
    `GroundingPromptBuilder` (§5.5), and an optional `LLMClient`. All can
    be injected; the retriever reuses the index's embedder by design.

    `llm_client` may be `None` for the offline legs: `retrieve_context()`
    and `build_prompt()` still work, while `answer()` raises a typed
    `ConfigurationError` instead of pretending a model exists.
    """

    def __init__(
        self,
        index: ChromaIndex,
        llm_client: LLMClientProtocol | None = None,
        embedder: object | None = None,
    ) -> None:
        if index._embedder is None:
            raise ConfigurationError(
                "Pipeline requires an index with an embedder; construct ChromaIndex(embedder=...)",
                code="embedder_not_configured",
            )
        # The retriever uses the index's embedder (§5.3.5); the pipeline
        # does not create a separate one, so query and chunk embeddings
        # stay in the same space. `embedder` is accepted for callers that
        # already hold one, but the index's own instance wins.
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
        """Retrieve and assemble evidence (§6.2); does not call the LLM.

        Returns the `GroundedContext` with citation mapping, chunk texts,
        and budget applied. A caller that only needs the evidence (e.g.
        an evaluation harness, CLI demonstration) can stop here; a caller
        that needs an answer passes the context through the prompt layer
        to the model (§5.5).
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
        history: Sequence[str] | None = None,
    ) -> list[BaseMessage]:
        """Full §6.2 flow through `GroundedContext`; does not call the LLM."""
        context = self.retrieve_context(
            query, filter_applied=filter_applied, k=k, budget_tokens=budget_tokens
        )
        return self._prompt_builder.build(query=query, ctx=context, history=history)

    def answer(
        self,
        query: str,
        *,
        filter_applied: dict[str, Any] | None = None,
        k: int = 20,
        budget_tokens: int = 3000,
        history: Sequence[str] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> Answer:
        """Generate an answer and verify every citation it makes.

        The returned `Answer` carries the model's text, the passages its
        ``[n]`` markers resolved to, and the checker's verdict. A faithful
        abstention and an uncited answer are returned with
        ``grounded=False``; a marker that names no retrieved chunk raises
        `ValidationError` (`invalid_citation`) because that answer presents
        invented evidence as verified.
        """
        if self._llm is None:
            raise ConfigurationError(
                "Answer generation requires an LLM client; construct "
                "GroundedAnswerPipeline(llm_client=...)",
                code="llm_not_configured",
            )

        context = self.retrieve_context(
            query, filter_applied=filter_applied, k=k, budget_tokens=budget_tokens
        )
        messages = self._prompt_builder.build(query=query, ctx=context, history=history)
        responses = self._llm.generate_messages(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = _response_text(responses)
        check = validate_citations(text, context)

        return Answer(
            query=query,
            text=text,
            citations=tuple(
                Citation.from_chunk(marker, context.citation_mapping[marker])
                for marker in check.markers
            ),
            grounded=check.grounded,
            abstained=check.abstained,
            reason=check.reason,
        )

    def stream(
        self,
        query: str,
        *,
        filter_applied: dict[str, Any] | None = None,
        k: int = 20,
        budget_tokens: int = 3000,
        history: Sequence[str] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """Stream assistant response chunks without blocking for the full text."""
        if self._llm is None:
            raise ConfigurationError(
                "Streaming requires an LLM client; construct "
                "GroundedAnswerPipeline(llm_client=...)",
                code="llm_not_configured",
            )
        context = self.retrieve_context(
            query, filter_applied=filter_applied, k=k, budget_tokens=budget_tokens
        )
        messages = self._prompt_builder.build(query=query, ctx=context, history=history)
        return self._llm.stream(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
