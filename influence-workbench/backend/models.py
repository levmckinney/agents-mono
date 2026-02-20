"""API request/response models."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class PairRole(str, Enum):
    train = "train"
    query = "query"
    both = "both"


class RunStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


# ---------------------------------------------------------------------------
# Pair models
# ---------------------------------------------------------------------------


class Pair(BaseModel):
    pair_id: str
    prompt: str
    completion: str
    role: PairRole = PairRole.both
    metadata: dict = {}


# ---------------------------------------------------------------------------
# Probe set models
# ---------------------------------------------------------------------------


class ProbeSetCreate(BaseModel):
    name: str
    pairs: list[Pair] = []


class ProbeSetUpdate(BaseModel):
    name: Optional[str] = None
    pairs: Optional[list[Pair]] = None


class ProbeSetSummary(BaseModel):
    id: str
    name: str
    pair_count: int
    created_at: str
    updated_at: str


class ProbeSetDetail(ProbeSetSummary):
    pairs: list[Pair]


# ---------------------------------------------------------------------------
# Run models
# ---------------------------------------------------------------------------


class RunSummary(BaseModel):
    id: str
    probe_set_id: str
    status: RunStatus
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None


class RunDetail(RunSummary):
    exit_code: Optional[int] = None
    error_message: Optional[str] = None
    config_snapshot: dict = {}


class RunResults(BaseModel):
    run_id: str
    query_results: list[dict]
    train_results: list[dict]
    influences: list[dict]


# ---------------------------------------------------------------------------
# Tool models — Infini-gram search
# ---------------------------------------------------------------------------


class SearchPretrainingRequest(BaseModel):
    completion: str
    max_docs: int = 10


class InfinigramDocSpan(BaseModel):
    text: str
    is_match: bool


class InfinigramDocument(BaseModel):
    doc_ix: int
    doc_len: int
    disp_len: int
    spans: list[InfinigramDocSpan]
    full_text: str


class SearchPretrainingResponse(BaseModel):
    documents: list[InfinigramDocument]
    query: str
    count: int


# ---------------------------------------------------------------------------
# Tool models — Span extraction
# ---------------------------------------------------------------------------


class ExtractSpanRequest(BaseModel):
    document_text: str
    match_start: int
    match_end: int
    span_length: int = 256


class ExtractSpanResponse(BaseModel):
    prompt: str
    completion: str


# ---------------------------------------------------------------------------
# Tool models — Claude context generation
# ---------------------------------------------------------------------------


class GenerateContextRequest(BaseModel):
    completion: str
    instruction: Optional[str] = None


class GenerateContextResponse(BaseModel):
    generated_prompt: str
    model: str


# ---------------------------------------------------------------------------
# Tool models — Bulk operations
# ---------------------------------------------------------------------------


class BulkRoleRequest(BaseModel):
    pair_ids: list[str]
    role: PairRole
