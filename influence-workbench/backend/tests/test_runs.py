"""Integration tests for run lifecycle with mocked subprocess."""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient

FIXTURES = Path(__file__).parent / "fixtures"


class MockProcess:
    """Mock async subprocess that copies fixture CSVs to the output dir."""

    def __init__(self, output_dir: str):
        self.returncode = 0
        self._output_dir = Path(output_dir)
        self._stderr_lines = [
            b"Loading model...\n",
            b"Computing influence scores...\n",
            b"Done.\n",
        ]
        self.stderr = self._make_stderr()

    async def _make_stderr(self):
        for line in self._stderr_lines:
            yield line

    async def wait(self):
        # Copy fixture CSVs to output dir
        self._output_dir.mkdir(parents=True, exist_ok=True)
        for csv_name in ("query.csv", "train.csv", "influences.csv"):
            src = FIXTURES / csv_name
            if src.exists():
                shutil.copy(src, self._output_dir / csv_name)
        return self.returncode


class MockFailedProcess:
    """Mock async subprocess that fails."""

    def __init__(self):
        self.returncode = 1
        self._stderr_lines = [
            b"Error: model not found\n",
        ]
        self.stderr = self._make_stderr()

    async def _make_stderr(self):
        for line in self._stderr_lines:
            yield line

    async def wait(self):
        return self.returncode


def _create_mock_subprocess(output_dir_holder: dict, fail: bool = False):
    """Create a mock for asyncio.create_subprocess_exec."""

    async def mock_exec(*args, **kwargs):
        if fail:
            return MockFailedProcess()
        # Extract output_dir from CLI args
        args_list = list(args)
        if "--output-dir" in args_list:
            idx = args_list.index("--output-dir")
            output_dir_holder["output_dir"] = args_list[idx + 1]
        return MockProcess(output_dir_holder.get("output_dir", "/tmp/test"))

    return mock_exec


@pytest.mark.asyncio
class TestRunLifecycle:
    async def _create_probe_set_with_pairs(self, client: AsyncClient) -> str:
        """Helper: create a probe set with valid train+query pairs."""
        resp = await client.post(
            "/api/probe-sets",
            json={
                "name": "Run Test Set",
                "pairs": [
                    {"pair_id": "q1", "prompt": "What is 1+1?", "completion": " 2.", "role": "query"},
                    {"pair_id": "t1", "prompt": "What is 2+2?", "completion": " 4.", "role": "train"},
                    {"pair_id": "b1", "prompt": "What is 3+3?", "completion": " 6.", "role": "both"},
                ],
            },
        )
        assert resp.status_code == 201
        return resp.json()["id"]

    async def test_create_run(self, client: AsyncClient):
        probe_set_id = await self._create_probe_set_with_pairs(client)
        output_dir_holder = {}

        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=_create_mock_subprocess(output_dir_holder),
        ):
            resp = await client.post(f"/api/probe-sets/{probe_set_id}/run")
            assert resp.status_code == 201
            data = resp.json()
            assert data["status"] == "pending"
            assert data["probe_set_id"] == probe_set_id
            run_id = data["id"]

            # Give the background task time to complete
            await asyncio.sleep(0.5)

        # Check run status is now completed
        resp = await client.get(f"/api/runs/{run_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert data["exit_code"] == 0

    async def test_create_run_probe_set_not_found(self, client: AsyncClient):
        resp = await client.post("/api/probe-sets/nonexistent/run")
        assert resp.status_code == 404

    async def test_create_run_no_pairs(self, client: AsyncClient):
        # Create empty probe set
        resp = await client.post(
            "/api/probe-sets", json={"name": "Empty Set"}
        )
        probe_set_id = resp.json()["id"]

        resp = await client.post(f"/api/probe-sets/{probe_set_id}/run")
        assert resp.status_code == 400

    async def test_create_run_missing_query_role(self, client: AsyncClient):
        resp = await client.post(
            "/api/probe-sets",
            json={
                "name": "Train Only",
                "pairs": [
                    {"pair_id": "t1", "prompt": "Q?", "completion": "A.", "role": "train"},
                ],
            },
        )
        probe_set_id = resp.json()["id"]

        resp = await client.post(f"/api/probe-sets/{probe_set_id}/run")
        assert resp.status_code == 400

    async def test_list_runs(self, client: AsyncClient):
        probe_set_id = await self._create_probe_set_with_pairs(client)
        output_dir_holder = {}

        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=_create_mock_subprocess(output_dir_holder),
        ):
            await client.post(f"/api/probe-sets/{probe_set_id}/run")
            await asyncio.sleep(0.3)

        resp = await client.get("/api/runs")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1

        # Filter by probe_set_id
        resp = await client.get(f"/api/runs?probe_set_id={probe_set_id}")
        assert resp.status_code == 200
        assert len(resp.json()) == 1

        resp = await client.get("/api/runs?probe_set_id=nonexistent")
        assert resp.status_code == 200
        assert len(resp.json()) == 0

    async def test_get_run_not_found(self, client: AsyncClient):
        resp = await client.get("/api/runs/nonexistent")
        assert resp.status_code == 404

    async def test_run_results(self, client: AsyncClient):
        probe_set_id = await self._create_probe_set_with_pairs(client)
        output_dir_holder = {}

        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=_create_mock_subprocess(output_dir_holder),
        ):
            resp = await client.post(f"/api/probe-sets/{probe_set_id}/run")
            run_id = resp.json()["id"]
            await asyncio.sleep(0.5)

        resp = await client.get(f"/api/runs/{run_id}/results")
        assert resp.status_code == 200
        data = resp.json()
        assert data["run_id"] == run_id
        assert len(data["query_results"]) == 2
        assert len(data["train_results"]) == 3
        assert len(data["influences"]) == 6

        # Verify structure
        assert data["query_results"][0]["query_id"] == "query_001"
        assert "loss" in data["query_results"][0]
        assert data["influences"][0]["query_id"] == "query_001"
        assert data["influences"][0]["train_id"] == "train_001"

    async def test_run_results_not_completed(self, client: AsyncClient):
        """Can't get results for a run that hasn't completed."""
        probe_set_id = await self._create_probe_set_with_pairs(client)

        # Create a run but don't let it complete - mock subprocess to hang
        async def hanging_exec(*args, **kwargs):
            proc = MagicMock()
            proc.returncode = None
            proc.stderr = AsyncIteratorMock([])

            async def wait_forever():
                await asyncio.sleep(100)
                return 0

            proc.wait = wait_forever
            return proc

        with patch("asyncio.create_subprocess_exec", side_effect=hanging_exec):
            resp = await client.post(f"/api/probe-sets/{probe_set_id}/run")
            run_id = resp.json()["id"]

        # Immediately try to get results (run is still pending/running)
        resp = await client.get(f"/api/runs/{run_id}/results")
        assert resp.status_code == 400

    async def test_failed_run(self, client: AsyncClient):
        probe_set_id = await self._create_probe_set_with_pairs(client)

        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=_create_mock_subprocess({}, fail=True),
        ):
            resp = await client.post(f"/api/probe-sets/{probe_set_id}/run")
            run_id = resp.json()["id"]
            await asyncio.sleep(0.5)

        resp = await client.get(f"/api/runs/{run_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "failed"
        assert data["exit_code"] == 1


class AsyncIteratorMock:
    """Helper to mock an async iterator."""
    def __init__(self, items):
        self._items = iter(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._items)
        except StopIteration:
            raise StopAsyncIteration
