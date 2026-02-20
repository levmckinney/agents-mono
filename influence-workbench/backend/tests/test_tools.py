"""Integration tests for tool endpoints (mocked external APIs)."""

from __future__ import annotations

import pytest


async def test_search_pretraining(client):
    resp = await client.post(
        "/api/search-pretraining",
        json={"completion": "matched text", "max_docs": 5},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["query"] == "matched text"
    assert data["count"] == 100
    assert len(data["documents"]) == 1
    doc = data["documents"][0]
    assert doc["doc_ix"] == 42
    assert len(doc["spans"]) == 3
    assert doc["spans"][1]["is_match"] is True
    assert doc["full_text"] == "Before the matched text after."


async def test_search_pretraining_no_client(client, app):
    app.state.infinigram_client = None
    resp = await client.post(
        "/api/search-pretraining",
        json={"completion": "test"},
    )
    assert resp.status_code == 503


async def test_extract_span(client):
    text = "First sentence. Second sentence. The target completion here."
    match_start = text.index("The target")
    match_end = len(text)
    resp = await client.post(
        "/api/extract-span",
        json={
            "document_text": text,
            "match_start": match_start,
            "match_end": match_end,
            "span_length": 256,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["completion"] == "The target completion here."
    assert len(data["prompt"]) > 0


async def test_extract_span_invalid_offsets(client):
    resp = await client.post(
        "/api/extract-span",
        json={
            "document_text": "short",
            "match_start": 10,
            "match_end": 20,
            "span_length": 100,
        },
    )
    assert resp.status_code == 400


async def test_extract_span_start_ge_end(client):
    resp = await client.post(
        "/api/extract-span",
        json={
            "document_text": "some text",
            "match_start": 5,
            "match_end": 5,
            "span_length": 100,
        },
    )
    assert resp.status_code == 400


async def test_generate_context(client):
    resp = await client.post(
        "/api/generate-context",
        json={"completion": "the answer is 42"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["generated_prompt"] == "This is a generated context prompt."
    assert data["model"] == "claude-haiku-4-5-20251001"


async def test_generate_context_with_instruction(client, app):
    resp = await client.post(
        "/api/generate-context",
        json={
            "completion": "the answer is 42",
            "instruction": "Write as a Wikipedia article",
        },
    )
    assert resp.status_code == 200
    # Verify the mock was called with instruction
    app.state.claude_client.generate_context.assert_called_with(
        "the answer is 42", instruction="Write as a Wikipedia article"
    )


async def test_generate_context_no_client(client, app):
    app.state.claude_client = None
    resp = await client.post(
        "/api/generate-context",
        json={"completion": "test"},
    )
    assert resp.status_code == 503
