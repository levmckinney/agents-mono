"""CRUD endpoints for saving and loading conversations."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from backend.models import ConversationDetail, ConversationSave, ConversationSummary

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/conversations", tags=["conversations"])


def _conversations_dir(request: Request) -> Path:
    """Return the conversations storage directory, creating it if needed."""
    data_dir = Path(request.app.state.config.data_dir)
    conv_dir = data_dir / "conversations"
    conv_dir.mkdir(parents=True, exist_ok=True)
    return conv_dir


def _safe_conversation_path(conv_dir: Path, conversation_id: str) -> Path:
    """Resolve a conversation file path, guarding against path traversal."""
    path = (conv_dir / f"{conversation_id}.json").resolve()
    if not path.is_relative_to(conv_dir.resolve()):
        raise HTTPException(status_code=400, detail="Invalid conversation ID")
    return path


def _read_conversation(path: Path) -> dict:
    """Read and parse a single conversation JSON file."""
    with open(path) as f:
        return json.load(f)


def _detail_from_data(data: dict) -> ConversationDetail:
    """Build a ConversationDetail from raw dict data."""
    return ConversationDetail(
        id=data["id"],
        name=data["name"],
        mode=data["mode"],
        conversation=data.get("conversation"),
        text=data.get("text"),
        system_prompt=data.get("system_prompt"),
        metadata=data.get("metadata"),
        created_at=data.get("metadata", {}).get(
            "created_at", data.get("created_at", "")
        ),
    )


def _summary_from_data(data: dict) -> ConversationSummary:
    """Build a ConversationSummary from raw dict data."""
    created_at = data.get("metadata", {}).get(
        "created_at", data.get("created_at", "")
    )
    turn_count = None
    char_count = None
    if data.get("mode") == "chat" and data.get("conversation"):
        turn_count = len(data["conversation"])
    elif data.get("mode") == "raw" and data.get("text"):
        char_count = len(data["text"])

    return ConversationSummary(
        id=data["id"],
        name=data["name"],
        mode=data["mode"],
        created_at=created_at,
        turn_count=turn_count,
        char_count=char_count,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("", response_model=ConversationDetail, status_code=201)
async def save_conversation(body: ConversationSave, request: Request):
    """Save a new conversation to disk."""
    conv_dir = _conversations_dir(request)
    conv_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    metadata = body.metadata or {}
    metadata.setdefault("created_at", now)
    metadata.setdefault("updated_at", now)

    data = {
        "id": conv_id,
        "name": body.name,
        "mode": body.mode,
        "metadata": metadata,
    }

    if body.mode == "chat":
        data["conversation"] = body.conversation or []
        if body.system_prompt is not None:
            data["system_prompt"] = body.system_prompt
    elif body.mode == "raw":
        data["text"] = body.text or ""
    else:
        raise HTTPException(status_code=400, detail=f"Unknown mode: {body.mode}")

    path = conv_dir / f"{conv_id}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info("Saved conversation %s (%s) to %s", conv_id, body.name, path)
    return _detail_from_data(data)


@router.get("", response_model=list[ConversationSummary])
async def list_conversations(request: Request):
    """List all saved conversations, sorted by created_at descending."""
    conv_dir = _conversations_dir(request)
    summaries: list[ConversationSummary] = []

    for path in conv_dir.glob("*.json"):
        try:
            data = _read_conversation(path)
            summaries.append(_summary_from_data(data))
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Skipping invalid conversation file %s: %s", path, exc)
            continue

    # Sort by created_at descending (most recent first)
    summaries.sort(key=lambda s: s.created_at, reverse=True)
    return summaries


@router.get("/export")
async def export_conversations(request: Request):
    """Export all conversations as a JSONL file download."""
    conv_dir = _conversations_dir(request)

    def generate_lines():
        for path in sorted(conv_dir.glob("*.json")):
            try:
                data = _read_conversation(path)
                yield json.dumps(data) + "\n"
            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning("Skipping %s during export: %s", path, exc)
                continue

    return StreamingResponse(
        generate_lines(),
        media_type="application/x-ndjson",
        headers={
            "Content-Disposition": "attachment; filename=conversations.jsonl"
        },
    )


@router.get("/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(conversation_id: str, request: Request):
    """Load a single saved conversation."""
    conv_dir = _conversations_dir(request)
    path = _safe_conversation_path(conv_dir, conversation_id)

    if not path.exists():
        raise HTTPException(status_code=404, detail="Conversation not found")

    try:
        data = _read_conversation(path)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=500, detail=f"Invalid conversation file: {exc}"
        )

    return _detail_from_data(data)


@router.delete("/{conversation_id}", status_code=204)
async def delete_conversation(conversation_id: str, request: Request):
    """Delete a saved conversation."""
    conv_dir = _conversations_dir(request)
    path = _safe_conversation_path(conv_dir, conversation_id)

    if not path.exists():
        raise HTTPException(status_code=404, detail="Conversation not found")

    path.unlink()
    logger.info("Deleted conversation %s", conversation_id)
