"""Tests for the /api/project endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.app import create_app


@pytest.fixture()
def mock_client(tmp_path):
    """Create a test client with mock mode enabled."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("mock_model: true\n")
    app = create_app(config_path=str(config_path))
    with TestClient(app) as client:
        yield client


class TestSchemas:
    """Verify Pydantic schemas import and validate correctly."""

    def test_token_projection(self):
        from backend.models import TokenProjection

        tp = TokenProjection(token_id=42, token_str="hello", projection=1.5, position=0)
        assert tp.token_id == 42
        assert tp.projection == 1.5

    def test_turn_span(self):
        from backend.models import TurnSpan

        span = TurnSpan(
            turn=0, role="user", start=0, end=5, text="hi", mean_projection=0.3,
        )
        assert span.role == "user"
        assert span.end == 5

    def test_projection_response(self):
        from backend.models import ProjectionResponse, TokenProjection

        resp = ProjectionResponse(
            tokens=[
                TokenProjection(
                    token_id=1, token_str="a", projection=0.5, position=0,
                )
            ],
            spans=[],
            layer=32,
            model_name="test",
        )
        assert resp.layer == 32
        assert len(resp.tokens) == 1

    def test_chat_request(self):
        from backend.models import ChatRequest

        req = ChatRequest(conversation=[{"role": "user", "content": "hello"}])
        assert len(req.conversation) == 1

    def test_raw_text_request(self):
        from backend.models import RawTextRequest

        req = RawTextRequest(text="some raw text")
        assert req.text == "some raw text"


class TestMockChatEndpoint:
    """Test POST /api/project/chat in mock mode."""

    def test_basic_response_shape(self, mock_client):
        resp = mock_client.post(
            "/api/project/chat",
            json={
                "conversation": [
                    {"role": "user", "content": "Hello there"},
                    {"role": "assistant", "content": "Hi how are you"},
                ],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "tokens" in data
        assert "spans" in data
        assert data["layer"] == 32
        assert data["model_name"] == "mock"

    def test_spans_match_conversation(self, mock_client):
        conversation = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "World"},
        ]
        resp = mock_client.post(
            "/api/project/chat", json={"conversation": conversation},
        )
        data = resp.json()
        assert len(data["spans"]) == 2
        assert data["spans"][0]["role"] == "user"
        assert data["spans"][1]["role"] == "assistant"

    def test_span_boundaries_are_contiguous(self, mock_client):
        conversation = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello world"},
            {"role": "assistant", "content": "Hi there friend"},
        ]
        resp = mock_client.post(
            "/api/project/chat", json={"conversation": conversation},
        )
        data = resp.json()
        spans = data["spans"]
        # Each span's start should equal the previous span's end
        for i in range(1, len(spans)):
            assert spans[i]["start"] == spans[i - 1]["end"]

    def test_token_count_matches_spans(self, mock_client):
        conversation = [
            {"role": "user", "content": "one two three"},
        ]
        resp = mock_client.post(
            "/api/project/chat", json={"conversation": conversation},
        )
        data = resp.json()
        total_span_tokens = sum(s["end"] - s["start"] for s in data["spans"])
        assert total_span_tokens == len(data["tokens"])

    def test_empty_conversation(self, mock_client):
        resp = mock_client.post(
            "/api/project/chat", json={"conversation": []},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["tokens"] == []
        assert data["spans"] == []


class TestMockRawEndpoint:
    """Test POST /api/project/raw in mock mode."""

    def test_basic_response_shape(self, mock_client):
        resp = mock_client.post(
            "/api/project/raw", json={"text": "Hello world foo bar"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "tokens" in data
        assert data["spans"] == []
        assert data["layer"] == 32
        assert data["model_name"] == "mock"

    def test_token_count(self, mock_client):
        text = "one two three four"
        resp = mock_client.post("/api/project/raw", json={"text": text})
        data = resp.json()
        # Mock splits on whitespace
        assert len(data["tokens"]) == 4

    def test_positions_are_sequential(self, mock_client):
        resp = mock_client.post(
            "/api/project/raw", json={"text": "a b c d e"},
        )
        data = resp.json()
        positions = [t["position"] for t in data["tokens"]]
        assert positions == list(range(len(positions)))


class TestHealthEndpoint:
    """Verify health check still works with the router included."""

    def test_health_mock(self, mock_client):
        resp = mock_client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "mock"
