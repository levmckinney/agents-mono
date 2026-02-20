"""Unit tests for InfinigramClient deduplication logic."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from backend.clients.infinigram import InfinigramClient, API_PAGE_SIZE, MAX_ATTEMPTS


def _make_response(doc_ixs: list[int], count: int = 1000) -> dict:
    """Build a fake search_docs API response."""
    return {
        "cnt": count,
        "documents": [
            {
                "doc_ix": ix,
                "doc_len": 500,
                "disp_len": 200,
                "spans": [["some text ", None], ["match", 0]],
            }
            for ix in doc_ixs
        ],
    }


def _make_client() -> InfinigramClient:
    client = InfinigramClient(
        http_client=None,  # type: ignore[arg-type]
        api_url="https://api.test/",
        index="test-index",
    )
    return client


@pytest.mark.asyncio
async def test_search_dedup_multiple_calls():
    """Overlapping batches yield the correct number of unique documents."""
    client = _make_client()

    # Call 1: docs 0-9, Call 2: docs 5-14 (5 overlap), Call 3: docs 10-19
    client._post = AsyncMock(
        side_effect=[
            _make_response(list(range(10))),
            _make_response(list(range(5, 15))),
            _make_response(list(range(10, 20))),
        ]
    )

    result = await client.search("test", max_docs=15)
    assert len(result["documents"]) == 15
    doc_ixs = [d["doc_ix"] for d in result["documents"]]
    assert len(set(doc_ixs)) == 15  # all unique
    assert result["count"] == 1000


@pytest.mark.asyncio
async def test_search_early_exit_all_dupes():
    """Exits early when a batch adds zero new documents."""
    client = _make_client()

    same_docs = _make_response([1, 2, 3], count=3)
    client._post = AsyncMock(return_value=same_docs)

    result = await client.search("rare query", max_docs=20)
    assert len(result["documents"]) == 3
    # First call gets 3 new, second call gets 0 new → exits
    assert client._post.call_count == 2


@pytest.mark.asyncio
async def test_search_max_attempts_cap():
    """Does not exceed MAX_ATTEMPTS API calls."""
    client = _make_client()

    call_count = 0

    async def always_new_docs(payload: dict) -> dict:
        nonlocal call_count
        start = call_count * 10
        call_count += 1
        return _make_response(list(range(start, start + 10)), count=10000)

    client._post = always_new_docs  # type: ignore[assignment]

    result = await client.search("test", max_docs=200)
    assert call_count <= MAX_ATTEMPTS
    assert len(result["documents"]) <= MAX_ATTEMPTS * API_PAGE_SIZE


@pytest.mark.asyncio
async def test_search_small_request_single_call():
    """Requesting <= 10 docs makes exactly one API call."""
    client = _make_client()
    client._post = AsyncMock(return_value=_make_response(list(range(5)), count=100))

    result = await client.search("test", max_docs=5)
    assert len(result["documents"]) == 5
    assert client._post.call_count == 1


@pytest.mark.asyncio
async def test_search_zero_count():
    """Zero-count response returns empty results with a single call."""
    client = _make_client()
    client._post = AsyncMock(return_value={"cnt": 0, "documents": []})

    result = await client.search("nonexistent", max_docs=10)
    assert result["documents"] == []
    assert result["count"] == 0
    assert client._post.call_count == 1


@pytest.mark.asyncio
async def test_search_parses_spans():
    """Documents have correctly parsed spans and full_text."""
    client = _make_client()
    client._post = AsyncMock(return_value=_make_response([42]))

    result = await client.search("test", max_docs=1)
    doc = result["documents"][0]
    assert doc["doc_ix"] == 42
    assert len(doc["spans"]) == 2
    assert doc["spans"][0] == {"text": "some text ", "is_match": False}
    assert doc["spans"][1] == {"text": "match", "is_match": True}
    assert doc["full_text"] == "some text match"
