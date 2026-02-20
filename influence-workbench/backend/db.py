"""SQLite schema and async helpers using aiosqlite."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import aiosqlite

SCHEMA = """
CREATE TABLE IF NOT EXISTS probe_sets (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    probe_set_id TEXT NOT NULL REFERENCES probe_sets(id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    exit_code INTEGER,
    error_message TEXT,
    config_snapshot TEXT NOT NULL
);
"""


async def init_db(db_path: str | Path) -> None:
    """Create tables if they don't exist."""
    async with aiosqlite.connect(db_path) as db:
        await db.executescript(SCHEMA)
        await db.execute("PRAGMA foreign_keys = ON")
        await db.commit()


async def get_connection(db_path: str | Path) -> aiosqlite.Connection:
    """Open a connection with foreign keys enabled."""
    db = await aiosqlite.connect(db_path)
    await db.execute("PRAGMA foreign_keys = ON")
    db.row_factory = aiosqlite.Row
    return db


# ---------------------------------------------------------------------------
# Probe set CRUD
# ---------------------------------------------------------------------------


async def insert_probe_set(
    db: aiosqlite.Connection, *, id: str, name: str, created_at: str, updated_at: str
) -> None:
    await db.execute(
        "INSERT INTO probe_sets (id, name, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (id, name, created_at, updated_at),
    )
    await db.commit()


async def list_probe_sets(db: aiosqlite.Connection) -> list[dict]:
    cursor = await db.execute(
        "SELECT id, name, created_at, updated_at FROM probe_sets ORDER BY created_at DESC"
    )
    rows = await cursor.fetchall()
    return [dict(row) for row in rows]


async def get_probe_set(db: aiosqlite.Connection, id: str) -> Optional[dict]:
    cursor = await db.execute(
        "SELECT id, name, created_at, updated_at FROM probe_sets WHERE id = ?", (id,)
    )
    row = await cursor.fetchone()
    return dict(row) if row else None


async def update_probe_set(
    db: aiosqlite.Connection,
    id: str,
    *,
    name: Optional[str] = None,
    updated_at: str,
) -> Optional[dict]:
    if name is not None:
        await db.execute(
            "UPDATE probe_sets SET name = ?, updated_at = ? WHERE id = ?",
            (name, updated_at, id),
        )
    else:
        await db.execute(
            "UPDATE probe_sets SET updated_at = ? WHERE id = ?",
            (updated_at, id),
        )
    await db.commit()
    return await get_probe_set(db, id)


async def delete_probe_set(db: aiosqlite.Connection, id: str) -> bool:
    cursor = await db.execute("DELETE FROM probe_sets WHERE id = ?", (id,))
    await db.commit()
    return cursor.rowcount > 0


# ---------------------------------------------------------------------------
# Run CRUD
# ---------------------------------------------------------------------------


async def insert_run(
    db: aiosqlite.Connection,
    *,
    id: str,
    probe_set_id: str,
    status: str,
    created_at: str,
    config_snapshot: str,
) -> None:
    await db.execute(
        "INSERT INTO runs (id, probe_set_id, status, created_at, config_snapshot) VALUES (?, ?, ?, ?, ?)",
        (id, probe_set_id, status, created_at, config_snapshot),
    )
    await db.commit()


async def list_runs(
    db: aiosqlite.Connection, *, probe_set_id: Optional[str] = None
) -> list[dict]:
    if probe_set_id:
        cursor = await db.execute(
            "SELECT * FROM runs WHERE probe_set_id = ? ORDER BY created_at DESC",
            (probe_set_id,),
        )
    else:
        cursor = await db.execute("SELECT * FROM runs ORDER BY created_at DESC")
    rows = await cursor.fetchall()
    return [dict(row) for row in rows]


async def get_run(db: aiosqlite.Connection, id: str) -> Optional[dict]:
    cursor = await db.execute("SELECT * FROM runs WHERE id = ?", (id,))
    row = await cursor.fetchone()
    return dict(row) if row else None


async def update_run_status(
    db: aiosqlite.Connection,
    id: str,
    *,
    status: str,
    started_at: Optional[str] = None,
    finished_at: Optional[str] = None,
    exit_code: Optional[int] = None,
    error_message: Optional[str] = None,
) -> None:
    fields = ["status = ?"]
    values: list = [status]
    if started_at is not None:
        fields.append("started_at = ?")
        values.append(started_at)
    if finished_at is not None:
        fields.append("finished_at = ?")
        values.append(finished_at)
    if exit_code is not None:
        fields.append("exit_code = ?")
        values.append(exit_code)
    if error_message is not None:
        fields.append("error_message = ?")
        values.append(error_message)
    values.append(id)
    await db.execute(
        f"UPDATE runs SET {', '.join(fields)} WHERE id = ?",
        values,
    )
    await db.commit()
