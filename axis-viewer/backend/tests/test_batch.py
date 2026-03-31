"""Tests for the /api/batch endpoints."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from backend.app import create_app


@pytest.fixture()
def mock_client(tmp_path):
    """Create a test client with mock mode enabled, using a temp data_dir."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"mock_model: true\ndata_dir: {tmp_path / 'data'}\n")
    app = create_app(config_path=str(config_path))
    with TestClient(app) as client:
        yield client


def _make_jsonl(conversations: list[dict]) -> bytes:
    """Build JSONL bytes from a list of conversation dicts."""
    return "\n".join(json.dumps(c) for c in conversations).encode("utf-8")


class TestBatchUpload:
    """Test POST /api/batch/upload."""

    def test_upload_chat_conversations(self, mock_client):
        content = _make_jsonl([
            {
                "name": "conv-1",
                "mode": "chat",
                "conversation": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi"},
                ],
            },
            {
                "name": "conv-2",
                "mode": "chat",
                "conversation": [
                    {"role": "user", "content": "Test"},
                ],
            },
        ])
        resp = mock_client.post(
            "/api/batch/upload",
            files={"file": ("test.jsonl", content, "application/x-ndjson")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2
        assert len(data["conversations"]) == 2
        assert data["batch_id"]
        # Check conversation summaries
        names = {c["name"] for c in data["conversations"]}
        assert names == {"conv-1", "conv-2"}

    def test_upload_raw_text(self, mock_client):
        content = _make_jsonl([
            {"name": "raw-1", "mode": "raw", "text": "Some raw text here"},
        ])
        resp = mock_client.post(
            "/api/batch/upload",
            files={"file": ("test.jsonl", content, "application/x-ndjson")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        assert data["conversations"][0]["mode"] == "raw"
        assert data["conversations"][0]["char_count"] == len("Some raw text here")

    def test_upload_mixed_modes(self, mock_client):
        content = _make_jsonl([
            {
                "name": "chat-one",
                "mode": "chat",
                "conversation": [{"role": "user", "content": "Hi"}],
            },
            {
                "name": "raw-one",
                "mode": "raw",
                "text": "Raw text",
            },
        ])
        resp = mock_client.post(
            "/api/batch/upload",
            files={"file": ("test.jsonl", content, "application/x-ndjson")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2
        modes = {c["mode"] for c in data["conversations"]}
        assert modes == {"chat", "raw"}

    def test_upload_invalid_json(self, mock_client):
        content = b"not valid json\n"
        resp = mock_client.post(
            "/api/batch/upload",
            files={"file": ("bad.jsonl", content, "application/x-ndjson")},
        )
        assert resp.status_code == 400
        assert "Invalid JSON" in resp.json()["detail"]

    def test_upload_invalid_mode(self, mock_client):
        content = _make_jsonl([
            {"name": "bad", "mode": "invalid_mode", "text": "x"},
        ])
        resp = mock_client.post(
            "/api/batch/upload",
            files={"file": ("test.jsonl", content, "application/x-ndjson")},
        )
        assert resp.status_code == 400
        assert "Invalid mode" in resp.json()["detail"]

    def test_upload_empty_file(self, mock_client):
        resp = mock_client.post(
            "/api/batch/upload",
            files={"file": ("empty.jsonl", b"", "application/x-ndjson")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 0

    def test_upload_default_name(self, mock_client):
        """Lines without a name get a default name."""
        content = _make_jsonl([
            {"mode": "chat", "conversation": [{"role": "user", "content": "test"}]},
        ])
        resp = mock_client.post(
            "/api/batch/upload",
            files={"file": ("test.jsonl", content, "application/x-ndjson")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["conversations"][0]["name"] == "conversation-1"


class TestBatchProject:
    """Test POST /api/batch/{batch_id}/project (SSE streaming)."""

    def _upload(self, client, conversations: list[dict]) -> str:
        """Helper: upload a batch and return the batch_id."""
        content = _make_jsonl(conversations)
        resp = client.post(
            "/api/batch/upload",
            files={"file": ("test.jsonl", content, "application/x-ndjson")},
        )
        assert resp.status_code == 200
        return resp.json()["batch_id"]

    def test_project_returns_sse_stream(self, mock_client):
        batch_id = self._upload(mock_client, [
            {
                "name": "conv-a",
                "mode": "chat",
                "conversation": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there"},
                ],
            },
        ])
        # Use stream=True to get raw SSE response
        with mock_client.stream("POST", f"/api/batch/{batch_id}/project") as resp:
            assert resp.status_code == 200
            events = []
            for line in resp.iter_lines():
                line = line.strip()
                if line.startswith("event:"):
                    events.append({"event": line[len("event:"):].strip()})
                elif line.startswith("data:") and events:
                    events[-1]["data"] = json.loads(line[len("data:"):].strip())

        # Should have progress, result, and done events
        event_types = [e["event"] for e in events]
        assert "progress" in event_types
        assert "result" in event_types
        assert "done" in event_types

    def test_project_result_has_projection_data(self, mock_client):
        batch_id = self._upload(mock_client, [
            {
                "name": "test-conv",
                "mode": "chat",
                "conversation": [
                    {"role": "user", "content": "Hello world"},
                    {"role": "assistant", "content": "Hi there"},
                ],
            },
        ])
        with mock_client.stream("POST", f"/api/batch/{batch_id}/project") as resp:
            result_events = []
            current_event = {}
            for line in resp.iter_lines():
                line = line.strip()
                if line.startswith("event:"):
                    current_event = {"event": line[len("event:"):].strip()}
                elif line.startswith("data:") and current_event:
                    current_event["data"] = json.loads(line[len("data:"):].strip())
                    if current_event["event"] == "result":
                        result_events.append(current_event)
                    current_event = {}

        assert len(result_events) == 1
        result_data = result_events[0]["data"]
        assert result_data["name"] == "test-conv"
        assert result_data["projection"] is not None
        assert result_data["projection"]["tokens"]
        assert result_data["projection"]["spans"]
        assert result_data["summary"] is not None
        assert "mean_projection" in result_data["summary"]
        assert "drift_amount" in result_data["summary"]

    def test_project_raw_text(self, mock_client):
        batch_id = self._upload(mock_client, [
            {"name": "raw-conv", "mode": "raw", "text": "Hello world foo bar"},
        ])
        with mock_client.stream("POST", f"/api/batch/{batch_id}/project") as resp:
            result_events = []
            current_event = {}
            for line in resp.iter_lines():
                line = line.strip()
                if line.startswith("event:"):
                    current_event = {"event": line[len("event:"):].strip()}
                elif line.startswith("data:") and current_event:
                    current_event["data"] = json.loads(line[len("data:"):].strip())
                    if current_event["event"] == "result":
                        result_events.append(current_event)
                    current_event = {}

        assert len(result_events) == 1
        result_data = result_events[0]["data"]
        assert result_data["mode"] == "raw"
        assert result_data["projection"]["spans"] == []
        # Raw mode: no drift
        assert result_data["summary"]["drift_amount"] is None

    def test_project_multiple_conversations(self, mock_client):
        batch_id = self._upload(mock_client, [
            {
                "name": f"conv-{i}",
                "mode": "chat",
                "conversation": [
                    {"role": "user", "content": f"Message {i}"},
                    {"role": "assistant", "content": f"Reply {i}"},
                ],
            }
            for i in range(5)
        ])
        with mock_client.stream("POST", f"/api/batch/{batch_id}/project") as resp:
            result_events = []
            progress_events = []
            current_event = {}
            for line in resp.iter_lines():
                line = line.strip()
                if line.startswith("event:"):
                    current_event = {"event": line[len("event:"):].strip()}
                elif line.startswith("data:") and current_event:
                    current_event["data"] = json.loads(line[len("data:"):].strip())
                    if current_event["event"] == "result":
                        result_events.append(current_event)
                    elif current_event["event"] == "progress":
                        progress_events.append(current_event)
                    current_event = {}

        assert len(result_events) == 5
        assert len(progress_events) == 5
        # All progress events should have total == 5
        for pe in progress_events:
            assert pe["data"]["total"] == 5

    def test_project_nonexistent_batch(self, mock_client):
        resp = mock_client.post("/api/batch/nonexistent-id/project")
        assert resp.status_code == 404


class TestBatchSchemas:
    """Verify batch-related Pydantic schemas."""

    def test_batch_upload_response(self):
        from backend.models import BatchUploadResponse, ConversationSummary

        resp = BatchUploadResponse(
            batch_id="abc123",
            count=1,
            conversations=[
                ConversationSummary(
                    id="c1", name="test", mode="chat",
                    created_at="2025-01-01", turn_count=2,
                )
            ],
        )
        assert resp.batch_id == "abc123"
        assert resp.count == 1

    def test_batch_conversation_result(self):
        from backend.models import BatchConversationResult, BatchConversationSummary

        result = BatchConversationResult(
            id="c1",
            name="test",
            mode="chat",
            summary=BatchConversationSummary(
                mean_projection=0.5,
                min_projection=-1.0,
                max_projection=2.0,
                drift_amount=1.5,
            ),
        )
        assert result.summary is not None
        assert result.summary.drift_amount == 1.5

    def test_batch_conversation_result_with_error(self):
        from backend.models import BatchConversationResult

        result = BatchConversationResult(
            id="c1",
            name="test",
            mode="chat",
            error="Something went wrong",
        )
        assert result.error == "Something went wrong"
        assert result.projection is None
