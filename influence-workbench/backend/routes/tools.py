"""Tool endpoints: pretraining search, span extraction, context generation."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from backend.models import (
    ExtractSpanRequest,
    ExtractSpanResponse,
    GenerateContextRequest,
    GenerateContextResponse,
    InfinigramDocument,
    InfinigramDocSpan,
    SearchPretrainingRequest,
    SearchPretrainingResponse,
)
from backend.span import extract_span

router = APIRouter(prefix="/api", tags=["tools"])


@router.post("/search-pretraining")
async def search_pretraining(
    body: SearchPretrainingRequest, request: Request
) -> SearchPretrainingResponse:
    client = getattr(request.app.state, "infinigram_client", None)
    if client is None:
        raise HTTPException(status_code=503, detail="Infini-gram client not configured")

    try:
        result = await client.search(body.completion, max_docs=body.max_docs)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Infini-gram API error: {exc}")

    documents = [
        InfinigramDocument(
            doc_ix=d["doc_ix"],
            doc_len=d["doc_len"],
            disp_len=d["disp_len"],
            spans=[InfinigramDocSpan(**s) for s in d["spans"]],
            full_text=d["full_text"],
        )
        for d in result["documents"]
    ]
    return SearchPretrainingResponse(
        documents=documents,
        query=result["query"],
        count=result["count"],
    )


@router.post("/extract-span")
async def extract_span_endpoint(body: ExtractSpanRequest) -> ExtractSpanResponse:
    if body.match_start < 0 or body.match_end > len(body.document_text):
        raise HTTPException(status_code=400, detail="match offsets out of range")
    if body.match_start >= body.match_end:
        raise HTTPException(status_code=400, detail="match_start must be < match_end")

    prompt, completion = extract_span(
        body.document_text,
        body.match_start,
        body.match_end,
        span_length=body.span_length,
    )
    return ExtractSpanResponse(prompt=prompt, completion=completion)


@router.post("/generate-context")
async def generate_context(
    body: GenerateContextRequest, request: Request
) -> GenerateContextResponse:
    client = getattr(request.app.state, "claude_client", None)
    if client is None:
        raise HTTPException(status_code=503, detail="Claude client not configured")

    try:
        generated = await client.generate_context(
            body.completion, instruction=body.instruction
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Claude API error: {exc}")

    return GenerateContextResponse(
        generated_prompt=generated,
        model=client.model,
    )
