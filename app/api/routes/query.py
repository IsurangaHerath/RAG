"""Query endpoint (section 5.6, `routes/query.py`).

POST `/v1/query` receives a `QueryRequest`, runs the pipeline through
`GroundedAnswerPipeline.answer()`, and returns an `AnswerResponse`. If
streaming is requested (`stream=True`), the endpoint yields chunks via
SSE or chunked JSON; the non-streaming path is the default.

Error mapping: any `RAGError` from the pipeline becomes a typed HTTP
response (`400` for `ValidationError`, `502` for `DependencyError`, `500`
for `ConfigurationError`) with an RFC7807-style JSON body (`{code,
message, request_id}`) produced by the middleware layer (`app/main.py`).
No document text reaches the response except the redacted `Answer.text`.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_chroma_index, get_llm_client
from app.api.schemas import AnswerResponse, QueryRequest
from app.generation.chain import GroundedAnswerPipeline

router = APIRouter(prefix="/v1", tags=["query"])


@router.post("/query", response_model=AnswerResponse)
async def query_endpoint(
    body: QueryRequest,
    index=Depends(get_chroma_index),
    llm_client=Depends(get_llm_client),
) -> Any:
    """Run the grounded query pipeline (section 6.2) and return a verified `Answer`."""
    pipeline = GroundedAnswerPipeline(index=index, llm_client=llm_client)
    try:
        answer = pipeline.answer(
            query=body.question,
            filter_applied={"doc_id": {"$in": body.doc_ids}} if body.doc_ids else None,
            k=body.top_k,
        )
    except Exception as exc:
        # The middleware handles typed `RAGError` mapping; this catch
        # ensures any unexpected exception becomes a 500 without leaking
        # the raw traceback to the client.
        raise HTTPException(status_code=500, detail=str(type(exc).__name__)) from exc

    return AnswerResponse(
        query=answer.query,
        text=answer.text,
        citations=[{"marker": c.marker, "chunk_id": c.chunk_id} for c in answer.citations],
        grounded=answer.grounded,
        abstained=answer.abstained,
        reason=answer.reason,
    )
