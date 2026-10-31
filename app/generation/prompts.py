"""Prompt assembly with grounding rules (§5.5) — Stage 37 (`generation/prompts.py`).

Not routed (§5.6): this is a protocol definition plus a concrete
implementation (`GroundingPromptBuilder`), not an endpoint. The system
prompt enforces the grounding contract described in `ARCHITECTURE.md`
§5.5: answer only from `<context>`; cite `[n]` markers for every factual
claim; when the context does not contain enough evidence, reply with the
fixed abstention phrase (`grounded=False`); never invent sources.

The concrete builder takes a `GroundedContext` (§5.4 → §5.5, Stage 36) —
which carries the citation mapping (`[1]` -> `chunk_id`), truncated chunk
texts with provenance annotations (`filename.pdf p.12 (doc_…:17)`), and
the budget applied (`budget_tokens`) — and produces a `list[BaseMessage]`
that the `LLMClient.generate_messages()` (§5.5, Stage 37, `llm_client.py`)
consumes.

No live model is called in this file; the concrete `build()` only
assembles strings. All provider interaction stays behind `LLMClient`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from app.domain.errors import ValidationError
from app.domain.models import GroundedContext

__all__ = ["GroundingPromptBuilder", "PromptBuilder"]

#: The fixed abstention phrase when the evidence set is insufficient.
ABSTENTION_PHRASE = (
    "I do not have enough information in the provided context to answer."
    " Please provide the relevant source passages."
)


class PromptBuilder(Protocol):
    """Turn a `GroundedContext` into a `list[BaseMessage]`."""

    def build(
        self,
        query: str,
        ctx: GroundedContext,
        history: Sequence[str] | None = None,
    ) -> list[BaseMessage]: ...


class GroundingPromptBuilder:
    """Concrete builder: system prompt with grounding rules + context blocks.

    The system message carries the rules (`ARCHITECTURE.md` §5.5): answer
    only from the context; cite `[n]` markers; abstain when evidence is
    missing. The user message includes the question and the evidence blocks
    in provenance order (filename, page, chunk id, full passage text) — the
    text is already budget-truncated by `SimpleContextBuilder`, so sending
    only a preview here would withhold the very sentences the model was
    asked to ground its answer in. Log/preview redaction lives in
    `GroundedContext.to_dict()` and `RetrievedChunk.summary()`, not here.

    The citation map is the authority for both the blocks and the legend:
    a marker in the prompt is exactly a marker `GroundingChecker` can
    resolve afterwards.
    """

    def build(
        self,
        query: str,
        ctx: GroundedContext,
        history: Sequence[str] | None = None,
    ) -> list[BaseMessage]:
        if not isinstance(query, str) or not query.strip():
            raise ValidationError(
                "prompt query must be a non-empty string",
                code="empty_embedding_input",
            )
        if not ctx.citation_mapping:
            # No evidence — the model must abstain rather than invent.
            system_content = (
                "You are a technical-document assistant. Answer ONLY from "
                "the provided context passages. Every factual claim must "
                "cite its source using `[n]` markers. If the context does "
                "not contain sufficient evidence, reply exactly with: "
                f'"{ABSTENTION_PHRASE}" (grounded=False). Never invent sources.'
            )
            user_content = (
                f"Question: {query}\n\n"
                f"Context: (empty — no passages retrieved)\n\n"
                f"Answer based only on the context above."
            )
        else:
            # Build context blocks in citation-marker order (§5.4 sketch:
            # filename.pdf p.12 (chunk_id) + passage text).
            ordered = sorted(ctx.citation_mapping.items())
            blocks = []
            for marker, chunk in ordered:
                provenance = f"{chunk.filename or 'unknown'}"
                if chunk.page is not None:
                    provenance += f" p.{chunk.page}"
                provenance += f" ({chunk.chunk_id})"
                blocks.append(f'[citation:{marker}] {provenance}\n"{chunk.content}"')
            context_text = "\n\n".join(blocks)
            citation_legend = ", ".join(
                f"[{marker}]: {chunk.chunk_id}" for marker, chunk in ordered
            )
            system_content = (
                "You are a technical-document assistant. Answer ONLY from "
                "the context passages below. Every factual claim must cite "
                "its source with `[n]` markers matching the citation legend. "
                "If the context does not contain sufficient evidence, reply "
                f'exactly with: "{ABSTENTION_PHRASE}" (grounded=False). '
                "Never invent sources, pages, or claims not present in the text."
            )
            user_content = (
                f"Question: {query}\n\n"
                f"Context passages:\n{context_text}\n\n"
                f"Citation legend: {citation_legend}\n\n"
                f"Answer using only the passages above. Cite sources with [n]."
            )
        messages: list[BaseMessage] = [
            SystemMessage(content=system_content),
            HumanMessage(content=user_content),
        ]
        if history is not None:
            # Simple history insertion (not a full turn-tracking layer —
            # that arrives with `chain.py`, Stage 40+).
            for turn in history:
                messages.insert(-1, HumanMessage(content=str(turn)))
        return messages
