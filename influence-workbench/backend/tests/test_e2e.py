"""End-to-end integration test using a fake run-query subprocess.

This test exercises the full pipeline: creating a probe set, launching a
run that spawns a real subprocess (fake_run_query.py), streaming stderr,
and parsing the resulting CSVs — without needing a GPU or pre-computed
hessian factors.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from httpx import AsyncClient

FIXTURES = Path(__file__).parent / "fixtures"
FAKE_SCRIPT = FIXTURES / "fake_run_query.py"


def _make_fake_subprocess_factory():
    """Return a patched create_subprocess_exec that runs fake_run_query.py.

    Replaces `uv run run-query <args>` with `python fake_run_query.py <args>`,
    keeping all other arguments identical so the runner's CLI construction is
    fully exercised.
    """
    real_create = asyncio.create_subprocess_exec

    async def fake_exec(*args, **kwargs):
        args_list = list(args)
        # Replace "uv", "run", "run-query" with python + fake script
        if len(args_list) >= 3 and args_list[:3] == ["uv", "run", "run-query"]:
            args_list = [sys.executable, str(FAKE_SCRIPT)] + args_list[3:]
        # Drop cwd since the fake script doesn't need to be in if-query dir
        kwargs.pop("cwd", None)
        return await real_create(*args_list, **kwargs)

    return fake_exec


@pytest.mark.asyncio
class TestEndToEnd:
    async def test_full_run_lifecycle(self, client: AsyncClient, app):
        """Create probe set -> launch run -> wait -> verify results."""

        # 1. Create a probe set with train + query pairs
        resp = await client.post(
            "/api/probe-sets",
            json={
                "name": "E2E Test Set",
                "pairs": [
                    {
                        "pair_id": "q1",
                        "prompt": "What is the capital of France?",
                        "completion": " The capital of France is Paris.",
                        "role": "query",
                    },
                    {
                        "pair_id": "q2",
                        "prompt": "What is 2+2?",
                        "completion": " 2+2 equals 4.",
                        "role": "query",
                    },
                    {
                        "pair_id": "t1",
                        "prompt": "Berlin is the capital of Germany.",
                        "completion": " That is correct.",
                        "role": "train",
                    },
                    {
                        "pair_id": "t2",
                        "prompt": "What is 3+3?",
                        "completion": " 3+3 equals 6.",
                        "role": "train",
                    },
                    {
                        "pair_id": "b1",
                        "prompt": "The sky is blue.",
                        "completion": " Yes, the sky appears blue.",
                        "role": "both",
                    },
                ],
            },
        )
        assert resp.status_code == 201
        probe_set_id = resp.json()["id"]
        assert resp.json()["pair_count"] == 5

        # 2. Launch a run using the fake subprocess
        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=_make_fake_subprocess_factory(),
        ):
            resp = await client.post(f"/api/probe-sets/{probe_set_id}/run")
            assert resp.status_code == 201
            run_data = resp.json()
            run_id = run_data["id"]
            assert run_data["status"] == "pending"
            assert run_data["probe_set_id"] == probe_set_id
            assert "config_snapshot" in run_data

            # 3. Wait for the background task to complete
            for _ in range(20):
                await asyncio.sleep(0.25)
                resp = await client.get(f"/api/runs/{run_id}")
                if resp.json()["status"] in ("completed", "failed"):
                    break

        # 4. Verify run completed successfully
        resp = await client.get(f"/api/runs/{run_id}")
        assert resp.status_code == 200
        run_detail = resp.json()
        assert run_detail["status"] == "completed", f"Run failed: {run_detail.get('error_message')}"
        assert run_detail["exit_code"] == 0

        # 5. Verify the run wrote train.json and query.json correctly
        data_dir = app.state.config.data_dir
        run_dir = Path(data_dir) / "probe_sets" / probe_set_id / "runs" / run_id
        assert (run_dir / "train.json").exists()
        assert (run_dir / "query.json").exists()

        import json

        with open(run_dir / "train.json") as f:
            train_data = json.load(f)
        with open(run_dir / "query.json") as f:
            query_data = json.load(f)

        # b1 (both) should appear in both lists
        train_ids = {p["pair_id"] for p in train_data}
        query_ids = {p["pair_id"] for p in query_data}
        assert train_ids == {"t1", "t2", "b1"}
        assert query_ids == {"q1", "q2", "b1"}

        # 6. Verify stderr.log was written
        assert (run_dir / "stderr.log").exists()
        log_content = (run_dir / "stderr.log").read_text()
        assert "Loading model" in log_content
        assert "Done." in log_content

        # 7. Fetch and verify results
        resp = await client.get(f"/api/runs/{run_id}/results")
        assert resp.status_code == 200
        results = resp.json()

        assert results["run_id"] == run_id
        assert len(results["query_results"]) == 3  # q1, q2, b1
        assert len(results["train_results"]) == 3  # t1, t2, b1
        assert len(results["influences"]) == 9  # 3 queries x 3 train

        # Verify query results structure
        query_ids_in_results = {r["query_id"] for r in results["query_results"]}
        assert query_ids_in_results == {"q1", "q2", "b1"}
        for qr in results["query_results"]:
            assert "loss" in qr
            assert "prompt" in qr
            assert "completion" in qr

        # Verify train results structure
        train_ids_in_results = {r["train_id"] for r in results["train_results"]}
        assert train_ids_in_results == {"t1", "t2", "b1"}

        # Verify influence scores structure
        for inf in results["influences"]:
            assert "query_id" in inf
            assert "train_id" in inf
            assert "influence_score" in inf
            assert isinstance(inf["influence_score"], (int, float))

        # 8. Verify run shows in list
        resp = await client.get("/api/runs")
        assert resp.status_code == 200
        all_runs = resp.json()
        assert len(all_runs) == 1
        assert all_runs[0]["id"] == run_id
        assert all_runs[0]["status"] == "completed"

        # Filter by probe_set_id
        resp = await client.get(f"/api/runs?probe_set_id={probe_set_id}")
        assert len(resp.json()) == 1

        resp = await client.get("/api/runs?probe_set_id=nonexistent")
        assert len(resp.json()) == 0

    async def test_run_with_failed_subprocess(self, client: AsyncClient):
        """Verify that a failing subprocess is properly tracked."""

        resp = await client.post(
            "/api/probe-sets",
            json={
                "name": "Fail Test",
                "pairs": [
                    {"pair_id": "q1", "prompt": "Q?", "completion": " A.", "role": "query"},
                    {"pair_id": "t1", "prompt": "T?", "completion": " T.", "role": "train"},
                ],
            },
        )
        probe_set_id = resp.json()["id"]

        # Capture real function before patching to avoid infinite recursion
        real_create = asyncio.create_subprocess_exec

        async def failing_exec(*args, **kwargs):
            return await real_create(
                sys.executable, "-c", "import sys; print('boom', file=sys.stderr); sys.exit(42)",
                **{k: v for k, v in kwargs.items() if k != "cwd"},
            )

        with patch("asyncio.create_subprocess_exec", side_effect=failing_exec):
            resp = await client.post(f"/api/probe-sets/{probe_set_id}/run")
            run_id = resp.json()["id"]

            for _ in range(20):
                await asyncio.sleep(0.25)
                resp = await client.get(f"/api/runs/{run_id}")
                if resp.json()["status"] in ("completed", "failed"):
                    break

        resp = await client.get(f"/api/runs/{run_id}")
        detail = resp.json()
        assert detail["status"] == "failed"
        assert detail["exit_code"] == 42
        assert "42" in detail["error_message"]
