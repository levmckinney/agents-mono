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


class GenerateRequest(BaseModel):
    conversation: list[dict]  # [{"role": "user", "content": "..."}, ...]
    temperature: float = 0.7
    max_new_tokens: int = 512


class GenerateResponse(BaseModel):
    content: str


# --- Conversation save/load schemas ---


class ConversationSave(BaseModel):
    name: str
    mode: str  # "chat" or "raw"
    conversation: list[dict] | None = None  # for chat mode
    text: str | None = None  # for raw mode
    system_prompt: str | None = None  # optional system prompt for chat mode
    metadata: dict | None = None


class ConversationSummary(BaseModel):
    id: str
    name: str
    mode: str
    created_at: str
    turn_count: int | None = None  # for chat mode
    char_count: int | None = None  # for raw mode


class ConversationDetail(BaseModel):
    id: str
    name: str
    mode: str
    conversation: list[dict] | None = None
    text: str | None = None
    system_prompt: str | None = None
    metadata: dict | None = None
    created_at: str
