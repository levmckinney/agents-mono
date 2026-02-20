"""CRUD endpoints for probe sets."""

from __future__ import annotations

import csv
import io
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, Response, UploadFile

from backend import db
from backend.models import (
    BulkRoleRequest,
    Pair,
    PairRole,
    ProbeSetCreate,
    ProbeSetDetail,
    ProbeSetUpdate,
    ProbeSetSummary,
)

router = APIRouter(prefix="/api/probe-sets", tags=["probe-sets"])


def _pairs_path(data_dir: str, probe_set_id: str) -> Path:
    return Path(data_dir) / "probe_sets" / probe_set_id / "pairs.json"


def _read_pairs(data_dir: str, probe_set_id: str) -> list[Pair]:
    path = _pairs_path(data_dir, probe_set_id)
    if not path.exists():
        return []
    with open(path) as f:
        return [Pair(**p) for p in json.load(f)]


def _write_pairs(data_dir: str, probe_set_id: str, pairs: list[Pair]) -> None:
    path = _pairs_path(data_dir, probe_set_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([p.model_dump() for p in pairs], f, indent=2)


def _make_detail(row: dict, pairs: list[Pair]) -> ProbeSetDetail:
    return ProbeSetDetail(
        id=row["id"],
        name=row["name"],
        pair_count=len(pairs),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        pairs=pairs,
    )


@router.post("", status_code=201)
async def create_probe_set(body: ProbeSetCreate, request: Request) -> ProbeSetDetail:
    conn = request.app.state.db
    data_dir = request.app.state.config.data_dir
    now = datetime.now(timezone.utc).isoformat()
    probe_set_id = uuid.uuid4().hex[:12]

    await db.insert_probe_set(
        conn, id=probe_set_id, name=body.name, created_at=now, updated_at=now
    )
    _write_pairs(data_dir, probe_set_id, body.pairs)

    row = await db.get_probe_set(conn, probe_set_id)
    return _make_detail(row, body.pairs)


@router.get("")
async def list_probe_sets(request: Request) -> list[ProbeSetSummary]:
    conn = request.app.state.db
    data_dir = request.app.state.config.data_dir
    rows = await db.list_probe_sets(conn)
    result = []
    for row in rows:
        pairs = _read_pairs(data_dir, row["id"])
        result.append(
            ProbeSetSummary(
                id=row["id"],
                name=row["name"],
                pair_count=len(pairs),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
        )
    return result


@router.get("/{probe_set_id}")
async def get_probe_set(probe_set_id: str, request: Request) -> ProbeSetDetail:
    conn = request.app.state.db
    data_dir = request.app.state.config.data_dir
    row = await db.get_probe_set(conn, probe_set_id)
    if not row:
        raise HTTPException(status_code=404, detail="Probe set not found")
    pairs = _read_pairs(data_dir, probe_set_id)
    return _make_detail(row, pairs)


@router.put("/{probe_set_id}")
async def update_probe_set(
    probe_set_id: str, body: ProbeSetUpdate, request: Request
) -> ProbeSetDetail:
    conn = request.app.state.db
    data_dir = request.app.state.config.data_dir
    row = await db.get_probe_set(conn, probe_set_id)
    if not row:
        raise HTTPException(status_code=404, detail="Probe set not found")

    now = datetime.now(timezone.utc).isoformat()
    await db.update_probe_set(conn, probe_set_id, name=body.name, updated_at=now)

    if body.pairs is not None:
        _write_pairs(data_dir, probe_set_id, body.pairs)

    updated_row = await db.get_probe_set(conn, probe_set_id)
    pairs = _read_pairs(data_dir, probe_set_id)
    return _make_detail(updated_row, pairs)


@router.delete("/{probe_set_id}", status_code=204)
async def delete_probe_set(probe_set_id: str, request: Request) -> Response:
    conn = request.app.state.db
    deleted = await db.delete_probe_set(conn, probe_set_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Probe set not found")
    # Note: pairs files left on disk for now; could clean up in production
    return Response(status_code=204)


@router.post("/{probe_set_id}/import-pairs")
async def import_pairs(
    probe_set_id: str, request: Request, file: UploadFile = File(...)
) -> ProbeSetDetail:
    """Import pairs from a CSV or JSON file, appending to existing pairs."""
    conn = request.app.state.db
    data_dir = request.app.state.config.data_dir
    row = await db.get_probe_set(conn, probe_set_id)
    if not row:
        raise HTTPException(status_code=404, detail="Probe set not found")

    content = await file.read()
    text = content.decode("utf-8")
    filename = file.filename or ""

    new_pairs: list[Pair] = []
    try:
        if filename.endswith(".json"):
            raw = json.loads(text)
            if not isinstance(raw, list):
                raise ValueError("JSON must be an array of pair objects")
            for item in raw:
                new_pairs.append(Pair(**item))
        else:
            # Default to CSV
            reader = csv.DictReader(io.StringIO(text))
            for row_data in reader:
                new_pairs.append(
                    Pair(
                        pair_id=row_data.get("pair_id", f"p{uuid.uuid4().hex[:8]}"),
                        prompt=row_data.get("prompt", ""),
                        completion=row_data.get("completion", ""),
                        role=PairRole(row_data.get("role", "both")),
                    )
                )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {exc}")

    existing_pairs = _read_pairs(data_dir, probe_set_id)
    all_pairs = existing_pairs + new_pairs
    _write_pairs(data_dir, probe_set_id, all_pairs)

    now = datetime.now(timezone.utc).isoformat()
    await db.update_probe_set(conn, probe_set_id, updated_at=now)

    updated_row = await db.get_probe_set(conn, probe_set_id)
    return _make_detail(updated_row, all_pairs)


@router.post("/{probe_set_id}/bulk-role")
async def bulk_set_role(
    probe_set_id: str, body: BulkRoleRequest, request: Request
) -> ProbeSetDetail:
    """Update role for specified pairs."""
    conn = request.app.state.db
    data_dir = request.app.state.config.data_dir
    row = await db.get_probe_set(conn, probe_set_id)
    if not row:
        raise HTTPException(status_code=404, detail="Probe set not found")

    pairs = _read_pairs(data_dir, probe_set_id)
    target_ids = set(body.pair_ids)
    for pair in pairs:
        if pair.pair_id in target_ids:
            pair.role = body.role

    _write_pairs(data_dir, probe_set_id, pairs)

    now = datetime.now(timezone.utc).isoformat()
    await db.update_probe_set(conn, probe_set_id, updated_at=now)

    updated_row = await db.get_probe_set(conn, probe_set_id)
    return _make_detail(updated_row, pairs)
