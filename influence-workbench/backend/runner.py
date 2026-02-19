"""Subprocess job runner for if-query."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from backend.contracts import IFQueryInputPair, IFQueryRunConfig
from backend.db import get_connection, update_run_status

if TYPE_CHECKING:
    pass


async def launch_run(
    app_state,
    *,
    run_id: str,
    probe_set_id: str,
    run_dir: Path,
    train_pairs: list[IFQueryInputPair],
    query_pairs: list[IFQueryInputPair],
) -> None:
    """Launch an if-query subprocess and manage its lifecycle.

    This coroutine is meant to be wrapped in asyncio.create_task so the
    calling endpoint returns immediately.
    """
    db_path = app_state.db_path
    config = app_state.config

    # 1. Write input JSON files
    train_json = run_dir / "train.json"
    query_json = run_dir / "query.json"
    train_json.write_text(
        json.dumps([p.model_dump() for p in train_pairs], indent=2)
    )
    query_json.write_text(
        json.dumps([p.model_dump() for p in query_pairs], indent=2)
    )

    # 2. Build run config
    results_dir = run_dir / "results"
    results_dir.mkdir(exist_ok=True)

    run_config = IFQueryRunConfig(
        model=config.model,
        revision=config.revision,
        factors_dir=config.factors_dir,
        train_json=str(train_json),
        query_json=str(query_json),
        output_dir=str(results_dir),
        query_batch_size=config.query_batch_size,
        train_batch_size=config.train_batch_size,
        max_length=config.max_length,
    )

    stderr_log_path = run_dir / "stderr.log"

    async with app_state.run_semaphore:
        conn = await get_connection(db_path)
        try:
            now = datetime.now(timezone.utc).isoformat()
            await update_run_status(conn, run_id, status="running", started_at=now)

            # 3. Spawn subprocess
            if_query_dir = str(Path(config.if_query_dir).resolve())
            cmd = ["uv", "run", "run-query"] + run_config.to_cli_args()

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
                cwd=if_query_dir,
            )

            # 4. Read stderr, write to log, broadcast to WebSocket subscribers
            with open(stderr_log_path, "w") as log_file:
                async for raw_line in proc.stderr:
                    line = raw_line.decode("utf-8", errors="replace").rstrip("\n")
                    log_file.write(line + "\n")
                    log_file.flush()

                    # Broadcast to WebSocket subscribers
                    subscribers = app_state.log_subscribers.get(run_id, set())
                    for queue in subscribers:
                        await queue.put(line)

            await proc.wait()

            # 5. Update run status
            now = datetime.now(timezone.utc).isoformat()
            if proc.returncode == 0:
                await update_run_status(
                    conn, run_id, status="completed", finished_at=now, exit_code=0
                )
            else:
                error_msg = f"Process exited with code {proc.returncode}"
                await update_run_status(
                    conn,
                    run_id,
                    status="failed",
                    finished_at=now,
                    exit_code=proc.returncode,
                    error_message=error_msg,
                )
        except Exception as e:
            now = datetime.now(timezone.utc).isoformat()
            await update_run_status(
                conn,
                run_id,
                status="failed",
                finished_at=now,
                error_message=str(e),
            )
        finally:
            # Notify WebSocket subscribers that the run is done
            subscribers = app_state.log_subscribers.get(run_id, set())
            for queue in subscribers:
                await queue.put(None)  # sentinel
            app_state.log_subscribers.pop(run_id, None)

            await conn.close()
