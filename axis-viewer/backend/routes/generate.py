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
    import torch

    probing_model = request.app.state.probing_model
    tokenizer = probing_model.tokenizer
    model = probing_model.model

    # Format conversation with chat template, requesting assistant generation
    formatted = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    prompt_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    new_tokens = outputs[0][prompt_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return text.strip()


@router.post("/generate", response_model=GenerateResponse)
async def generate(body: GenerateRequest, request: Request):
    """Generate an assistant response for the given conversation."""
    config = request.app.state.config

    if config.mock_model:
        return GenerateResponse(content=MOCK_RESPONSE)

    if request.app.state.probing_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    async with request.app.state.gpu_lock:
        content = await asyncio.to_thread(
            _generate_response,
            request,
            body.conversation,
            body.temperature,
            body.max_new_tokens,
        )
    return GenerateResponse(content=content)
