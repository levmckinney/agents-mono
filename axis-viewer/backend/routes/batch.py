"""Batch upload and projection endpoints."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, UploadFile
from sse_starlette.sse import EventSourceResponse

from backend.models import (
    BatchConversationResult,
    BatchConversationSummary,
    BatchUploadResponse,
    ConversationSummary,
    ProjectionResponse,
)
from backend.routes.project import (
    _extract_chat,
    _extract_raw,
    _mock_chat_response,
    _mock_raw_response,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/batch", tags=["batch"])


def _conversations_dir(request: Request) -> Path:
    """Return the conversations storage directory, creating it if needed."""
    data_dir = Path(request.app.state.config.data_dir)
    conv_dir = data_dir / "conversations"
    conv_dir.mkdir(parents=True, exist_ok=True)
    return conv_dir


def _compute_summary(
    projection: ProjectionResponse, mode: str
) -> BatchConversationSummary:
    """Compute summary statistics from projection data."""
    if not projection.tokens:
        return BatchConversationSummary(
            mean_projection=0.0,
            min_projection=0.0,
            max_projection=0.0,
            drift_amount=None,
        )

    all_projs = [t.projection for t in projection.tokens]
    mean_val = sum(all_projs) / len(all_projs)
    min_val = min(all_projs)
    max_val = max(all_projs)

    drift = None
    if mode == "chat" and projection.spans:
        span_means = [s.mean_projection for s in projection.spans]
        drift = max(span_means) - min(span_means)

    return BatchConversationSummary(
        mean_projection=mean_val,
        min_projection=min_val,
        max_projection=max_val,
        drift_amount=drift,
    )


@router.post("/upload", response_model=BatchUploadResponse)
async def upload_batch(file: UploadFile, request: Request):
    """Upload a JSONL file containing multiple conversations.

    Each line should be a JSON object with ``name``, ``mode``, and either
    ``conversation`` (for chat) or ``text`` (for raw) fields.
    """
    conv_dir = _conversations_dir(request)
    batch_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    content = await file.read()
    lines = content.decode("utf-8").strip().splitlines()

    summaries: list[ConversationSummary] = []
    for line_num, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid JSON on line {line_num}: {exc}",
            )

        name = obj.get("name", f"conversation-{line_num}")
        mode = obj.get("mode", "chat")
        if mode not in ("chat", "raw"):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid mode '{mode}' on line {line_num}",
            )

        conv_id = str(uuid.uuid4())
        metadata = obj.get("metadata", {})
        metadata.setdefault("created_at", now)
        metadata.setdefault("updated_at", now)
        metadata["batch_id"] = batch_id

        data: dict = {
            "id": conv_id,
            "name": name,
            "mode": mode,
            "metadata": metadata,
        }

        if mode == "chat":
            data["conversation"] = obj.get("conversation", [])
            if obj.get("system_prompt"):
                data["system_prompt"] = obj["system_prompt"]
        else:
            data["text"] = obj.get("text", "")

        path = conv_dir / f"{conv_id}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        turn_count = None
        char_count = None
        if mode == "chat":
            turn_count = len(data.get("conversation", []))
        else:
            char_count = len(data.get("text", ""))

        summaries.append(
            ConversationSummary(
                id=conv_id,
                name=name,
                mode=mode,
                created_at=metadata["created_at"],
                turn_count=turn_count,
                char_count=char_count,
            )
        )

    logger.info(
        "Batch %s uploaded: %d conversations from %s",
        batch_id,
        len(summaries),
        file.filename,
    )
    return BatchUploadResponse(
        batch_id=batch_id,
        count=len(summaries),
        conversations=summaries,
    )


@router.post("/{batch_id}/project")
async def project_batch(batch_id: str, request: Request):
    """Project all conversations in a batch, streaming progress via SSE.

    Events:
    - ``progress``: ``{"conversation_id", "index", "total"}`` as each starts
    - ``result``: ``{"conversation_id", "result": BatchConversationResult}``
    - ``done``: sent when all projections are complete
    """
    conv_dir = _conversations_dir(request)
    config = request.app.state.config

    # Collect conversations belonging to this batch
    batch_convs: list[dict] = []
    for path in conv_dir.glob("*.json"):
        try:
            with open(path) as f:
                data = json.load(f)
            if data.get("metadata", {}).get("batch_id") == batch_id:
                batch_convs.append(data)
        except (json.JSONDecodeError, KeyError):
            continue

    if not batch_convs:
        raise HTTPException(status_code=404, detail="Batch not found")

    # Sort by name for consistent ordering
    batch_convs.sort(key=lambda d: d.get("name", ""))
    total = len(batch_convs)

    async def event_generator():
        for idx, conv_data in enumerate(batch_convs):
            conv_id = conv_data["id"]
            name = conv_data["name"]
            mode = conv_data["mode"]

            # Send progress event
            yield {
                "event": "progress",
                "data": json.dumps(
                    {"conversation_id": conv_id, "index": idx, "total": total}
                ),
            }

            try:
                if mode == "chat":
                    conversation = conv_data.get("conversation", [])
                    if config.mock_model:
                        projection = _mock_chat_response(conversation)
                    else:
                        projection = await asyncio.to_thread(
                            _extract_chat, request, conversation
                        )
                else:
                    text = conv_data.get("text", "")
                    if config.mock_model:
                        projection = _mock_raw_response(text)
                    else:
                        projection = await asyncio.to_thread(
                            _extract_raw, request, text
                        )

                summary = _compute_summary(projection, mode)
                result = BatchConversationResult(
                    id=conv_id,
                    name=name,
                    mode=mode,
                    projection=projection,
                    summary=summary,
                )
            except Exception as exc:
                logger.exception(
                    "Error projecting conversation %s in batch %s",
                    conv_id,
                    batch_id,
                )
                result = BatchConversationResult(
                    id=conv_id,
                    name=name,
                    mode=mode,
                    error=str(exc),
                )

            yield {
                "event": "result",
                "data": result.model_dump_json(),
            }

        yield {"event": "done", "data": json.dumps({"status": "complete"})}

    return EventSourceResponse(event_generator())
