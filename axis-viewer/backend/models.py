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


# --- Model schemas ---


class ModelInfo(BaseModel):
    model_name: str
    target_layer: int
    total_layers: int
    short_name: str
    is_base: bool = False
    has_capping: bool = False
    status: str  # "loaded", "available", "switching"


class ModelSwitchRequest(BaseModel):
    model_name: str


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


# --- Batch schemas ---


class BatchUploadResponse(BaseModel):
    batch_id: str
    count: int
    conversations: list[ConversationSummary]


class BatchConversationSummary(BaseModel):
    """Summary statistics computed from a projection result."""

    mean_projection: float
    min_projection: float
    max_projection: float
    drift_amount: float | None = None  # max - min of per-turn means (chat only)


class BatchConversationResult(BaseModel):
    id: str
    name: str
    mode: str
    projection: ProjectionResponse | None = None
    error: str | None = None
    summary: BatchConversationSummary | None = None
