"""Run management and results endpoints."""

from __future__ import annotations

import asyncio
import csv
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect

from backend import db
from backend.contracts import IFQueryInfluenceRow, IFQueryQueryResult, IFQueryTrainResult
from backend.models import (
    RunDetail,
    RunResults,
    RunStatus,
    RunSummary,
)
from backend.pairs import split_pairs_by_role, validate_pairs_for_run
from backend.routes.probe_sets import _read_pairs

router = APIRouter(tags=["runs"])


def _run_dir(data_dir: str, probe_set_id: str, run_id: str) -> Path:
    return Path(data_dir) / "probe_sets" / probe_set_id / "runs" / run_id


def _run_to_summary(row: dict) -> RunSummary:
    return RunSummary(
        id=row["id"],
        probe_set_id=row["probe_set_id"],
        status=RunStatus(row["status"]),
        created_at=row["created_at"],
        started_at=row.get("started_at"),
        finished_at=row.get("finished_at"),
    )


def _run_to_detail(row: dict) -> RunDetail:
    config_snapshot = row.get("config_snapshot", "{}")
    if isinstance(config_snapshot, str):
        config_snapshot = json.loads(config_snapshot)
    return RunDetail(
        id=row["id"],
        probe_set_id=row["probe_set_id"],
        status=RunStatus(row["status"]),
        created_at=row["created_at"],
        started_at=row.get("started_at"),
        finished_at=row.get("finished_at"),
        exit_code=row.get("exit_code"),
        error_message=row.get("error_message"),
        config_snapshot=config_snapshot,
    )


@router.post("/api/probe-sets/{probe_set_id}/run", status_code=201)
async def create_run(probe_set_id: str, request: Request) -> RunDetail:
    conn = request.app.state.db
    config = request.app.state.config
    data_dir = config.data_dir

    # Verify probe set exists
    ps_row = await db.get_probe_set(conn, probe_set_id)
    if not ps_row:
        raise HTTPException(status_code=404, detail="Probe set not found")

    # Read and validate pairs
    pairs = _read_pairs(data_dir, probe_set_id)
    errors = validate_pairs_for_run(pairs)
    if errors:
        raise HTTPException(status_code=400, detail={"validation_errors": errors})

    # Split pairs
    train_pairs, query_pairs = split_pairs_by_role(pairs)

    # Create run record
    run_id = uuid.uuid4().hex[:12]
    now = datetime.now(timezone.utc).isoformat()
    run_path = _run_dir(data_dir, probe_set_id, run_id)
    run_path.mkdir(parents=True, exist_ok=True)
    (run_path / "results").mkdir(exist_ok=True)

    from backend.contracts import IFQueryRunConfig

    run_config = IFQueryRunConfig(
        model=config.model,
        revision=config.revision,
        factors_dir=str(Path(config.factors_dir).resolve()),
        train_json=str((run_path / "train.json").resolve()),
        query_json=str((run_path / "query.json").resolve()),
        output_dir=str((run_path / "results").resolve()),
        query_batch_size=config.query_batch_size,
        train_batch_size=config.train_batch_size,
        max_length=config.max_length,
    )

    config_snapshot = json.loads(run_config.model_dump_json())

    await db.insert_run(
        conn,
        id=run_id,
        probe_set_id=probe_set_id,
        status="pending",
        created_at=now,
        config_snapshot=json.dumps(config_snapshot),
    )

    # Launch subprocess in background
    from backend.runner import launch_run

    asyncio.create_task(
        launch_run(
            request.app.state,
            run_id=run_id,
            probe_set_id=probe_set_id,
            run_dir=run_path,
            train_pairs=train_pairs,
            query_pairs=query_pairs,
        )
    )

    row = await db.get_run(conn, run_id)
    return _run_to_detail(row)


@router.get("/api/runs")
async def list_runs(request: Request, probe_set_id: str | None = None) -> list[RunSummary]:
    conn = request.app.state.db
    rows = await db.list_runs(conn, probe_set_id=probe_set_id)
    return [_run_to_summary(row) for row in rows]


@router.get("/api/runs/{run_id}")
async def get_run(run_id: str, request: Request) -> RunDetail:
    conn = request.app.state.db
    row = await db.get_run(conn, run_id)
    if not row:
        raise HTTPException(status_code=404, detail="Run not found")
    return _run_to_detail(row)


@router.get("/api/runs/{run_id}/results")
async def get_run_results(run_id: str, request: Request) -> RunResults:
    conn = request.app.state.db
    row = await db.get_run(conn, run_id)
    if not row:
        raise HTTPException(status_code=404, detail="Run not found")

    if row["status"] != "completed":
        raise HTTPException(status_code=400, detail="Run has not completed successfully")

    config_snapshot = json.loads(row["config_snapshot"]) if isinstance(row["config_snapshot"], str) else row["config_snapshot"]
    output_dir = Path(config_snapshot["output_dir"])

    query_results = []
    train_results = []
    influences = []

    query_csv = output_dir / "query.csv"
    if query_csv.exists():
        with open(query_csv, newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                query_results.append(IFQueryQueryResult(**r).model_dump())

    train_csv = output_dir / "train.csv"
    if train_csv.exists():
        with open(train_csv, newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                train_results.append(IFQueryTrainResult(**r).model_dump())

    influences_csv = output_dir / "influences.csv"
    if influences_csv.exists():
        with open(influences_csv, newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                influences.append(IFQueryInfluenceRow(**r).model_dump())

    return RunResults(
        run_id=run_id,
        query_results=query_results,
        train_results=train_results,
        influences=influences,
    )


@router.websocket("/api/runs/{run_id}/logs")
async def stream_logs(websocket: WebSocket, run_id: str):
    await websocket.accept()
    app_state = websocket.app.state

    # Create a queue for this subscriber
    queue: asyncio.Queue = asyncio.Queue()
    if run_id not in app_state.log_subscribers:
        app_state.log_subscribers[run_id] = set()
    app_state.log_subscribers[run_id].add(queue)

    try:
        # Send existing log content if available
        conn = app_state.db
        row = await db.get_run(conn, run_id)
        if row:
            config_snapshot = json.loads(row["config_snapshot"]) if isinstance(row["config_snapshot"], str) else row["config_snapshot"]
            stderr_log = Path(config_snapshot["output_dir"]).parent / "stderr.log"
            if stderr_log.exists():
                with open(stderr_log) as f:
                    for line in f:
                        await websocket.send_text(line.rstrip("\n"))

            # If run is already done, close after sending history
            if row["status"] in ("completed", "failed"):
                await websocket.send_text(f"[run {row['status']}]")
                return

        # Stream new lines
        while True:
            line = await queue.get()
            if line is None:  # sentinel for "run finished"
                break
            await websocket.send_text(line)
    except WebSocketDisconnect:
        pass
    finally:
        app_state.log_subscribers.get(run_id, set()).discard(queue)
