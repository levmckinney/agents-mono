"""Subprocess job runner for if-query."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from backend.contracts import IFQueryInputPair, IFQueryRunConfig
from backend.db import get_connection, update_run_status

if TYPE_CHECKING:
    pass

# 10 MB — enough for progress bars and long model-loading lines
STREAM_LIMIT = 10 * 1024 * 1024


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
        factors_dir=str(Path(config.factors_dir).resolve()),
        train_json=str(train_json.resolve()),
        query_json=str(query_json.resolve()),
        output_dir=str(results_dir.resolve()),
        query_batch_size=config.query_batch_size,
        train_batch_size=config.train_batch_size,
        max_length=config.max_length,
    )

    stderr_log_path = run_dir / "stderr.log"
    stdout_log_path = run_dir / "stdout.log"

    async with app_state.run_semaphore:
        conn = await get_connection(db_path)
        try:
            now = datetime.now(timezone.utc).isoformat()
            await update_run_status(conn, run_id, status="running", started_at=now)

            # 3. Spawn subprocess
            if_query_dir = str(Path(config.if_query_dir).resolve())
            cmd = ["uv", "run", "run-query"] + run_config.to_cli_args()

            # Strip VIRTUAL_ENV so uv uses the if-query project's own venv
            env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=if_query_dir,
                env=env,
                limit=STREAM_LIMIT,
            )

            # 4. Read stdout and stderr concurrently, write to logs,
            #    broadcast stderr to WebSocket subscribers.
            #    tqdm writes \r to overwrite progress bars in-place;
            #    we split on \r and send only the last segment, prefixed
            #    with \r so the frontend can replace the previous line.
            async def _drain_stream(stream, log_path, broadcast=False):
                with open(log_path, "w") as log_file:
                    async for raw_line in stream:
                        text = raw_line.decode("utf-8", errors="replace").rstrip("\n")
                        # Split on \r — tqdm produces lines like
                        # "\rProgress 50%\rProgress 51%"
                        segments = text.split("\r")
                        # Write the final state to the log file
                        final = segments[-1] if segments[-1] else (segments[-2] if len(segments) > 1 else "")
                        if final:
                            log_file.write(final + "\n")
                            log_file.flush()
                        if broadcast:
                            subscribers = app_state.log_subscribers.get(run_id, set())
                            for seg in segments:
                                if not seg:
                                    continue
                                # Prefix with \r if this is a carriage-return update
                                msg = f"\r{seg}" if len(segments) > 1 else seg
                                for queue in subscribers:
                                    await queue.put(msg)

            await asyncio.gather(
                _drain_stream(proc.stdout, stdout_log_path),
                _drain_stream(proc.stderr, stderr_log_path, broadcast=True),
            )

            await proc.wait()

            # 5. Update run status
            now = datetime.now(timezone.utc).isoformat()
            if proc.returncode == 0:
                await update_run_status(
                    conn, run_id, status="completed", finished_at=now, exit_code=0
                )
            else:
                # Build an informative error message from the log files
                error_msg = _build_error_message(
                    proc.returncode, stderr_log_path, stdout_log_path
                )
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
            # Try to include log tail in the error message
            error_msg = str(e)
            if stderr_log_path.exists():
                tail = _tail_file(stderr_log_path, 20)
                if tail:
                    error_msg = f"{e}\n\n--- stderr tail ---\n{tail}"
            await update_run_status(
                conn,
                run_id,
                status="failed",
                finished_at=now,
                error_message=error_msg,
            )
        finally:
            # Notify WebSocket subscribers that the run is done
            subscribers = app_state.log_subscribers.get(run_id, set())
            for queue in subscribers:
                await queue.put(None)  # sentinel
            app_state.log_subscribers.pop(run_id, None)

            await conn.close()


def _tail_file(path: Path, n: int = 20) -> str:
    """Return the last n lines of a file."""
    try:
        lines = path.read_text().strip().splitlines()
        return "\n".join(lines[-n:])
    except Exception:
        return ""


def _build_error_message(
    returncode: int, stderr_path: Path, stdout_path: Path
) -> str:
    """Build an informative error message from process output."""
    parts = [f"Process exited with code {returncode}"]

    stderr_tail = _tail_file(stderr_path, 30)
    if stderr_tail:
        parts.append(f"\n--- stderr (last 30 lines) ---\n{stderr_tail}")

    stdout_tail = _tail_file(stdout_path, 10)
    if stdout_tail:
        parts.append(f"\n--- stdout (last 10 lines) ---\n{stdout_tail}")

    return "\n".join(parts)
