"""Projection endpoints: project token activations onto the Assistant Axis."""

from __future__ import annotations

import asyncio
import logging
import random

from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Request

if TYPE_CHECKING:
    import torch

from backend.models import (
    ChatRequest,
    ProjectionResponse,
    RawTextRequest,
    TokenProjection,
    TurnSpan,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/project", tags=["project"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _project_tokens(
    activations: torch.Tensor,
    axis: torch.Tensor,
    layer: int,
) -> list[float]:
    """Project each token's activation onto the axis at *layer*.

    Parameters
    ----------
    activations:
        Tensor of shape ``(num_tokens, hidden_size)`` – activations at the
        target layer for every token.
    axis:
        Tensor of shape ``(num_layers, hidden_size)``.
    layer:
        Which layer index to use from *axis*.

    Returns
    -------
    List of scalar projection values, one per token.
    """
    import torch as _torch

    axis_vec = axis[layer].float()
    axis_norm = _torch.norm(axis_vec)
    if axis_norm == 0:
        return [0.0] * activations.shape[0]
    axis_vec = axis_vec / axis_norm
    acts = activations.float()
    projections = (acts @ axis_vec).tolist()
    return projections


def _build_token_projections(
    token_ids: list[int],
    projections: list[float],
    tokenizer,
) -> list[TokenProjection]:
    """Build a list of ``TokenProjection`` from ids, projections, and a tokenizer."""
    tokens: list[TokenProjection] = []
    for i, (tid, proj) in enumerate(zip(token_ids, projections)):
        token_str = tokenizer.decode([tid])
        tokens.append(
            TokenProjection(
                token_id=tid,
                token_str=token_str,
                projection=proj,
                position=i,
            )
        )
    return tokens


def _build_turn_spans(
    spans: list[dict],
    projections: list[float],
) -> list[TurnSpan]:
    """Compute per-span mean projections and return ``TurnSpan`` objects."""
    result: list[TurnSpan] = []
    for span in spans:
        start = span["start"]
        end = span["end"]
        span_projs = projections[start:end]
        mean_proj = sum(span_projs) / len(span_projs) if span_projs else 0.0
        result.append(
            TurnSpan(
                turn=span["turn"],
                role=span["role"],
                start=start,
                end=end,
                text=span["text"],
                mean_projection=mean_proj,
            )
        )
    return result


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

def _mock_tokens_from_words(words: list[str], offset: int = 0) -> list[TokenProjection]:
    """Build mock ``TokenProjection`` objects by treating each word as a token."""
    return [
        TokenProjection(
            token_id=offset + i,
            token_str=word,
            projection=random.uniform(-2.0, 2.0),
            position=offset + i,
        )
        for i, word in enumerate(words)
    ]


def _mock_chat_response(conversation: list[dict]) -> ProjectionResponse:
    """Return synthetic data when the model is not loaded."""
    tokens: list[TokenProjection] = []
    spans: list[TurnSpan] = []
    pos = 0
    for turn_idx, msg in enumerate(conversation):
        role = msg.get("role") or "user"
        content = msg.get("content") or ""
        words = content.split() or ["<empty>"]
        span_tokens = _mock_tokens_from_words(words, offset=pos)
        span_projs = [t.projection for t in span_tokens]
        mean_proj = sum(span_projs) / len(span_projs) if span_projs else 0.0
        spans.append(
            TurnSpan(
                turn=turn_idx,
                role=role,
                start=pos,
                end=pos + len(span_tokens),
                text=content,
                mean_projection=mean_proj,
            )
        )
        tokens.extend(span_tokens)
        pos += len(span_tokens)
    return ProjectionResponse(
        tokens=tokens,
        spans=spans,
        layer=32,
        model_name="mock",
    )


def _mock_raw_response(text: str) -> ProjectionResponse:
    """Return synthetic data for raw text when the model is not loaded."""
    words = text.split() or ["<empty>"]
    return ProjectionResponse(
        tokens=_mock_tokens_from_words(words),
        spans=[],
        layer=32,
        model_name="mock",
    )


# ---------------------------------------------------------------------------
# Blocking extraction functions (run via asyncio.to_thread)
# ---------------------------------------------------------------------------

def _build_spans_from_template(tokenizer, conversation: list[dict]) -> tuple[list[int], list[dict]]:
    """Build token IDs and turn spans by comparing per-turn tokenizations.

    This avoids relying on assistant_axis's build_turn_spans which may be
    incompatible with transformers 5.x (returns dict instead of list).
    """
    # Tokenize the full conversation
    full_result = tokenizer.apply_chat_template(
        conversation, tokenize=True, add_generation_prompt=False,
    )
    # Handle transformers 5.x dict return
    if isinstance(full_result, dict):
        full_ids = full_result["input_ids"]
    else:
        full_ids = full_result

    # Build spans by tokenizing incrementally
    spans = []
    prev_len = 0
    for turn_idx, msg in enumerate(conversation):
        partial = conversation[:turn_idx + 1]
        partial_result = tokenizer.apply_chat_template(
            partial, tokenize=True, add_generation_prompt=False,
        )
        if isinstance(partial_result, dict):
            partial_ids = partial_result["input_ids"]
        else:
            partial_ids = partial_result

        start = prev_len
        end = len(partial_ids)
        spans.append({
            "turn": turn_idx,
            "role": msg.get("role", "unknown"),
            "start": start,
            "end": end,
            "text": msg.get("content", ""),
        })
        prev_len = end

    return full_ids, spans


def _extract_chat(request: Request, conversation: list[dict]) -> ProjectionResponse:
    """Extract activations for a chat conversation and project onto axis.

    This function performs GPU work and should be called via
    ``asyncio.to_thread``.
    """
    config = request.app.state.config
    extractor = request.app.state.extractor
    axis = request.app.state.axis
    target_layer = config.target_layer
    tokenizer = request.app.state.probing_model.tokenizer

    if not conversation:
        return ProjectionResponse(tokens=[], spans=[], layer=target_layer, model_name=config.model_name)

    # Build token IDs and spans ourselves (compatible with transformers 5.x)
    token_ids, spans = _build_spans_from_template(tokenizer, conversation)

    # Extract activations at the target layer
    activations = extractor.full_conversation(
        conversation,
        layer=target_layer,
        chat_format=True,
    )
    # activations shape: (num_tokens, hidden_size)

    if len(token_ids) != activations.shape[0]:
        logger.warning(
            "Token count mismatch: spans=%d, activations=%d. Using activation count.",
            len(token_ids), activations.shape[0],
        )
        # Truncate/pad token_ids to match activations
        if len(token_ids) > activations.shape[0]:
            token_ids = token_ids[:activations.shape[0]]

    projections = _project_tokens(activations, axis, target_layer)

    token_projs = _build_token_projections(token_ids, projections, tokenizer)
    turn_spans = _build_turn_spans(spans, projections)

    return ProjectionResponse(
        tokens=token_projs,
        spans=turn_spans,
        layer=target_layer,
        model_name=config.model_name,
    )


def _extract_raw(request: Request, text: str) -> ProjectionResponse:
    """Extract activations for raw text (no chat template) and project onto axis.

    This function performs GPU work and should be called via
    ``asyncio.to_thread``.
    """
    config = request.app.state.config
    extractor = request.app.state.extractor
    axis = request.app.state.axis
    target_layer = config.target_layer

    if not text.strip():
        return ProjectionResponse(tokens=[], spans=[], layer=target_layer, model_name=config.model_name)

    # full_conversation accepts a plain string with chat_format=False
    activations = extractor.full_conversation(
        text,
        layer=target_layer,
        chat_format=False,
    )
    # activations shape: (num_tokens, hidden_size)

    tokenizer = request.app.state.probing_model.tokenizer
    # Use the same tokenization path as full_conversation (add_special_tokens=False)
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    if len(token_ids) != activations.shape[0]:
        logger.warning(
            "Raw text token mismatch: encode=%d, activations=%d",
            len(token_ids), activations.shape[0],
        )
        # Activations may include BOS; truncate token_ids to match
        if len(token_ids) > activations.shape[0]:
            token_ids = token_ids[:activations.shape[0]]

    projections = _project_tokens(activations, axis, target_layer)
    token_projs = _build_token_projections(token_ids, projections, tokenizer)

    return ProjectionResponse(
        tokens=token_projs,
        spans=[],
        layer=target_layer,
        model_name=config.model_name,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/chat", response_model=ProjectionResponse)
async def project_chat(body: ChatRequest, request: Request):
    """Project a multi-turn conversation onto the Assistant Axis."""
    config = request.app.state.config

    if config.mock_model:
        return _mock_chat_response(body.conversation)

    if request.app.state.extractor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    async with request.app.state.gpu_lock:
        return await asyncio.to_thread(_extract_chat, request, body.conversation)


@router.post("/raw", response_model=ProjectionResponse)
async def project_raw(body: RawTextRequest, request: Request):
    """Project raw text (no chat template) onto the Assistant Axis."""
    config = request.app.state.config

    if config.mock_model:
        return _mock_raw_response(body.text)

    if request.app.state.extractor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    async with request.app.state.gpu_lock:
        return await asyncio.to_thread(_extract_raw, request, body.text)
