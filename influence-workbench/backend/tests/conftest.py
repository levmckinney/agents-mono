"""Shared fixtures for backend tests."""

from __future__ import annotations

import os
import tempfile

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from backend.app import create_app
from backend.db import get_connection, init_db


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Provide a temporary data directory."""
    return tmp_path


@pytest_asyncio.fixture
async def app(tmp_data_dir):
    """Create a test app with a temporary DB and data directory."""
    # Write a minimal config
    config_path = tmp_data_dir / "config.yaml"
    config_path.write_text(
        f"data_dir: {tmp_data_dir}\n"
        f'if_query_dir: "../if-query"\n'
        f'factors_dir: "/tmp/factors"\n'
    )

    test_app = create_app(config_path=str(config_path))

    # Manually run lifespan for testing
    async with test_app.router.lifespan_context(test_app):
        yield test_app


@pytest_asyncio.fixture
async def client(app):
    """Provide an async HTTP client for testing."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
