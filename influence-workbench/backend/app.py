"""FastAPI application with lifespan management."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI

from backend.clients.claude import ClaudeClient
from backend.clients.infinigram import InfinigramClient
from backend.config import WorkbenchConfig, load_config
from backend.db import get_connection, init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load config, init DB, create semaphore, init API clients."""
    config = load_config(app.state.config_path)
    data_dir = Path(config.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    db_path = data_dir / "probe_sets.db"
    await init_db(db_path)
    conn = await get_connection(db_path)

    http_client = httpx.AsyncClient()

    app.state.config = config
    app.state.db = conn
    app.state.db_path = str(db_path)
    app.state.run_semaphore = asyncio.Semaphore(config.max_concurrent_runs)
    app.state.log_subscribers = {}  # run_id -> set of asyncio.Queue
    app.state.http_client = http_client
    app.state.infinigram_client = InfinigramClient(
        http_client=http_client,
        api_url=config.infinigram_api_url,
        index=config.infinigram_index,
        max_attempts=config.infinigram_max_attempts,
    )
    app.state.claude_client = ClaudeClient(model=config.claude_model)

    yield

    await http_client.aclose()
    await conn.close()


def create_app(config_path: str = "config.yaml") -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(title="Influence Workbench", lifespan=lifespan)
    app.state.config_path = config_path

    from backend.routes.probe_sets import router as probe_sets_router
    from backend.routes.runs import router as runs_router
    from backend.routes.tools import router as tools_router

    app.include_router(probe_sets_router)
    app.include_router(runs_router)
    app.include_router(tools_router)

    return app


app = create_app()


def main():
    """Run the application with uvicorn."""
    import uvicorn

    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
