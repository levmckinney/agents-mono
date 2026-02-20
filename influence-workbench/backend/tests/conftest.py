"""Shared fixtures for backend tests."""

from __future__ import annotations

import os
import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from backend.app import create_app
from backend.db import get_connection, init_db


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Provide a temporary data directory."""
    return tmp_path


@pytest.fixture
def mock_infinigram_client():
    """Provide a mock InfinigramClient."""
    client = MagicMock()
    client.search = AsyncMock(
        return_value={
            "documents": [
                {
                    "doc_ix": 42,
                    "doc_len": 1000,
                    "disp_len": 500,
                    "spans": [
                        {"text": "Before the ", "is_match": False},
                        {"text": "matched text", "is_match": True},
                        {"text": " after.", "is_match": False},
                    ],
                    "full_text": "Before the matched text after.",
                }
            ],
            "query": "matched text",
            "count": 100,
        }
    )
    return client


@pytest.fixture
def mock_claude_client():
    """Provide a mock ClaudeClient."""
    client = MagicMock()
    client.model = "claude-haiku-4-5-20251001"
    client.generate_context = AsyncMock(
        return_value="This is a generated context prompt."
    )
    return client


@pytest_asyncio.fixture
async def app(tmp_data_dir, mock_infinigram_client, mock_claude_client):
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
        # Override with mocks after lifespan sets real clients
        test_app.state.infinigram_client = mock_infinigram_client
        test_app.state.claude_client = mock_claude_client
        yield test_app


@pytest_asyncio.fixture
async def client(app):
    """Provide an async HTTP client for testing."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
