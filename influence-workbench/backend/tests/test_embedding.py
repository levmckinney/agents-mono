"""Tests for the t-SNE embedding endpoint."""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest
from httpx import AsyncClient

from backend.tests.test_runs import MockProcess, _AsyncLineIter, _create_mock_subprocess

FIXTURES = Path(__file__).parent / "fixtures"


async def _create_completed_run(client: AsyncClient) -> str:
    """Create a probe set, launch a run, and wait for it to complete."""
    resp = await client.post(
        "/api/probe-sets",
        json={
            "name": "Embedding Test Set",
            "pairs": [
                {"pair_id": "q1", "prompt": "What is 1+1?", "completion": " 2.", "role": "query"},
                {"pair_id": "t1", "prompt": "What is 2+2?", "completion": " 4.", "role": "train"},
                {"pair_id": "b1", "prompt": "What is 3+3?", "completion": " 6.", "role": "both"},
            ],
        },
    )
    assert resp.status_code == 201
    probe_set_id = resp.json()["id"]

    output_dir_holder: dict = {}
    with patch(
        "asyncio.create_subprocess_exec",
        side_effect=_create_mock_subprocess(output_dir_holder),
    ):
        resp = await client.post(f"/api/probe-sets/{probe_set_id}/run")
        assert resp.status_code == 201
        run_id = resp.json()["id"]
        await asyncio.sleep(0.5)

    return run_id


@pytest.mark.asyncio
class TestEmbedding:
    async def test_embedding_returns_correct_shape(self, client: AsyncClient):
        run_id = await _create_completed_run(client)
        resp = await client.get(f"/api/runs/{run_id}/embedding")
        assert resp.status_code == 200
        data = resp.json()
        assert data["run_id"] == run_id
        # Fixture has 2 query results
        assert len(data["points"]) == 2
        assert data["n_query"] == 2
        assert data["n_train"] == 3
        for pt in data["points"]:
            assert "x" in pt
            assert "y" in pt
            assert "pair_id" in pt
            assert "prompt_preview" in pt
            assert "completion_preview" in pt
            assert "role" in pt

    async def test_embedding_all_query_ids_present(self, client: AsyncClient):
        run_id = await _create_completed_run(client)

        results_resp = await client.get(f"/api/runs/{run_id}/results")
        query_ids = {q["query_id"] for q in results_resp.json()["query_results"]}

        embed_resp = await client.get(f"/api/runs/{run_id}/embedding")
        embed_ids = {p["pair_id"] for p in embed_resp.json()["points"]}
        assert embed_ids == query_ids

    async def test_embedding_cached(self, client: AsyncClient):
        run_id = await _create_completed_run(client)

        resp1 = await client.get(f"/api/runs/{run_id}/embedding")
        assert resp1.status_code == 200

        # Second call should return the same data (from cache)
        resp2 = await client.get(f"/api/runs/{run_id}/embedding")
        assert resp2.status_code == 200
        assert resp1.json() == resp2.json()

    async def test_embedding_not_found(self, client: AsyncClient):
        resp = await client.get("/api/runs/nonexistent/embedding")
        assert resp.status_code == 404

    async def test_embedding_not_completed(self, client: AsyncClient):
        """Embedding for a non-completed run should return 400."""
        resp = await client.post(
            "/api/probe-sets",
            json={
                "name": "Pending Set",
                "pairs": [
                    {"pair_id": "q1", "prompt": "Q?", "completion": "A.", "role": "query"},
                    {"pair_id": "t1", "prompt": "Q?", "completion": "A.", "role": "train"},
                    {"pair_id": "b1", "prompt": "Q?", "completion": "A.", "role": "both"},
                ],
            },
        )
        probe_set_id = resp.json()["id"]

        # Create a hanging run
        async def hanging_exec(*args, **kwargs):
            from unittest.mock import MagicMock

            proc = MagicMock()
            proc.returncode = None
            proc.stdout = _AsyncLineIter([])
            proc.stderr = _AsyncLineIter([])

            async def wait_forever():
                await asyncio.sleep(100)
                return 0

            proc.wait = wait_forever
            return proc

        with patch("asyncio.create_subprocess_exec", side_effect=hanging_exec):
            resp = await client.post(f"/api/probe-sets/{probe_set_id}/run")
            run_id = resp.json()["id"]

        resp = await client.get(f"/api/runs/{run_id}/embedding")
        assert resp.status_code == 400

    async def test_embedding_single_query(self, client: AsyncClient):
        """With a single query pair, the point should be at (0, 0)."""
        # Create probe set with one both-role pair (produces 1 query)
        resp = await client.post(
            "/api/probe-sets",
            json={
                "name": "Single Pair Set",
                "pairs": [
                    {"pair_id": "b1", "prompt": "What is 1+1?", "completion": " 2.", "role": "both"},
                ],
            },
        )
        probe_set_id = resp.json()["id"]

        # Create a mock subprocess that writes single-pair CSVs
        # (instead of copying fixture files which have 2 queries)
        async def single_pair_exec(*args, **kwargs):
            args_list = list(args)
            output_dir = "/tmp/test"
            if "--output-dir" in args_list:
                idx = args_list.index("--output-dir")
                output_dir = args_list[idx + 1]
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            (output_path / "query.csv").write_text(
                "query_id,prompt,completion,loss\nb1,What is 1+1?, 2.,5.0\n"
            )
            (output_path / "train.csv").write_text(
                "train_id,prompt,completion\nb1,What is 1+1?, 2.\n"
            )
            (output_path / "influences.csv").write_text(
                "query_id,train_id,influence_score\nb1,b1,1000000.0\n"
            )

            from unittest.mock import MagicMock

            proc = MagicMock()
            proc.returncode = 0
            proc.stdout = _AsyncLineIter([])
            proc.stderr = _AsyncLineIter([b"Done.\n"])

            async def noop_wait():
                return 0

            proc.wait = noop_wait
            return proc

        with patch("asyncio.create_subprocess_exec", side_effect=single_pair_exec):
            resp = await client.post(f"/api/probe-sets/{probe_set_id}/run")
            run_id = resp.json()["id"]
            await asyncio.sleep(0.5)

        resp = await client.get(f"/api/runs/{run_id}/embedding")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["points"]) == 1
        assert data["points"][0]["x"] == 0.0
        assert data["points"][0]["y"] == 0.0
