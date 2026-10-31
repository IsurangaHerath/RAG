"""Context assembly: retrieval results -> grounded evidence (section 5.4 -> section 5.5, Stage 36).

`ContextBuilder.build()` takes a user question and the `RetrievedChunk`
results from `MMRRetriever.retrieve()` (Stage 31) and produces a
`GroundedContext` (section 5.5): ordered chunk blocks (provenance annotations:
filename.pdf p.12 (chunk_id) + truncated text), citation mapping
(`[1]` -> chunk), and the token budget applied. The chunks are ordered
by provenance (filename, page, section from `Chunk.metadata`) rather
than by retrieval score — citation numbering must be stable regardless of
MMR diversity selection.

The budget truncation is approximate (character-based approximation of
token budget) rather than exact token counting; exact budget management
is the generation layer's later refinement (section 5.5). The important contract
is that truncation is explicit (`GroundedContext.truncated = True`) and
that every cited chunk exists in the context (`citation_mapping` covers
all markers used in the chunks list).
"""

from collections.abc import Sequence

from app.domain.models import GroundedContext, RetrievedChunk

__all__ = ["SimpleContextBuilder"]


class SimpleContextBuilder:
    """Basic concrete `ContextBuilder`: order by provenance, truncate to budget.

    Not a Protocol declaration (`ContextBuilder` Protocol lives in
    `ARCHITECTURE.md` section 5.4); this is the first concrete implementation,
    sufficient for the integration stage (section 6.2) and the CLI demonstration.
    A full `ContextBuilder` with token-aware truncation and citation
    extraction arrives with `PromptBuilder` (section 5.5, Stage 37+).
    """

    def build(
        self,
        query: str,
        chunks: Sequence[RetrievedChunk],
        *,
        budget_tokens: int = 3000,
    ) -> GroundedContext:
        # Order by provenance: filename, then page (if present), then
        # section (if present), then chunk index — so citations are stable.
        ordered = sorted(
            chunks,
            key=lambda c: (
                c.filename or "",
                c.page if c.page is not None else 0,
                c.section or "",
                c.chunk_id,
            ),
        )
        # Approximate truncation: assume ~4 chars per token (English average);
        # exact token counting is the generation layer's job (section 5.5).
        budget_chars = budget_tokens * 4
        truncated = False
        selected: list[RetrievedChunk] = []
        total_chars = 0
        for chunk in ordered:
            if total_chars + chunk.char_count > budget_chars and selected:
                truncated = True
                break
            selected.append(chunk)
            total_chars += chunk.char_count

        # Build citation mapping: each chunk gets the next available number.
        citation_mapping = {i + 1: chunk for i, chunk in enumerate(selected)}

        return GroundedContext(
            query=query,
            chunks=selected,
            citation_mapping=citation_mapping,
            budget_tokens=budget_tokens,
            truncated=truncated,
        )
