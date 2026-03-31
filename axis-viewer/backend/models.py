"""Pydantic request/response schemas for the axis-viewer API."""

from __future__ import annotations

from pydantic import BaseModel


class TokenProjection(BaseModel):
    token_id: int
    token_str: str
    projection: float  # dot(activation, axis) / norm(axis) at target layer
    position: int  # index in token sequence


class TurnSpan(BaseModel):
    turn: int
    role: str  # "user" | "assistant" | "system"
    start: int  # token index (inclusive)
    end: int  # token index (exclusive)
    text: str
    mean_projection: float  # mean projection across tokens in this span


class ProjectionResponse(BaseModel):
    tokens: list[TokenProjection]
    spans: list[TurnSpan]  # empty list for raw text mode
    layer: int
    model_name: str


class ChatRequest(BaseModel):
    conversation: list[dict]  # [{"role": "user", "content": "..."}, ...]


class RawTextRequest(BaseModel):
    text: str
