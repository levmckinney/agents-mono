"""Generation endpoint: generate assistant responses using ProbingModel."""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Request

from backend.models import GenerateRequest, GenerateResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["generate"])


MOCK_RESPONSE = "This is a mock assistant response."


def _generate_response(
    request: Request,
    conversation: list[dict],
    temperature: float,
    max_new_tokens: int,
) -> str:
    """Run generation on the model. Blocking -- call via asyncio.to_thread."""
    probing_model = request.app.state.probing_model
    return probing_model.generate(
        conversation,
        enable_thinking=False,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )


@router.post("/generate", response_model=GenerateResponse)
async def generate(body: GenerateRequest, request: Request):
    """Generate an assistant response for the given conversation."""
    config = request.app.state.config

    if config.mock_model:
        return GenerateResponse(content=MOCK_RESPONSE)

    if request.app.state.probing_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    content = await asyncio.to_thread(
        _generate_response,
        request,
        body.conversation,
        body.temperature,
        body.max_new_tokens,
    )
    return GenerateResponse(content=content)
