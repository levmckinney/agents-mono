"""Integration tests for probe set CRUD endpoints."""

from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
class TestProbeSetCRUD:
    async def test_create_probe_set(self, client: AsyncClient):
        resp = await client.post(
            "/api/probe-sets",
            json={
                "name": "Test Set",
                "pairs": [
                    {
                        "pair_id": "p1",
                        "prompt": "Hello?",
                        "completion": "Hi.",
                        "role": "both",
                    }
                ],
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "Test Set"
        assert data["pair_count"] == 1
        assert len(data["pairs"]) == 1
        assert "id" in data

    async def test_list_probe_sets(self, client: AsyncClient):
        # Create two probe sets
        await client.post("/api/probe-sets", json={"name": "Set A"})
        await client.post("/api/probe-sets", json={"name": "Set B"})

        resp = await client.get("/api/probe-sets")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        names = {s["name"] for s in data}
        assert names == {"Set A", "Set B"}

    async def test_get_probe_set(self, client: AsyncClient):
        create_resp = await client.post(
            "/api/probe-sets",
            json={
                "name": "Get Test",
                "pairs": [
                    {
                        "pair_id": "p1",
                        "prompt": "Q?",
                        "completion": "A.",
                        "role": "train",
                    }
                ],
            },
        )
        probe_set_id = create_resp.json()["id"]

        resp = await client.get(f"/api/probe-sets/{probe_set_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Get Test"
        assert data["pair_count"] == 1
        assert data["pairs"][0]["pair_id"] == "p1"
        assert data["pairs"][0]["role"] == "train"

    async def test_get_probe_set_not_found(self, client: AsyncClient):
        resp = await client.get("/api/probe-sets/nonexistent")
        assert resp.status_code == 404

    async def test_update_probe_set_name(self, client: AsyncClient):
        create_resp = await client.post(
            "/api/probe-sets", json={"name": "Original"}
        )
        probe_set_id = create_resp.json()["id"]

        resp = await client.put(
            f"/api/probe-sets/{probe_set_id}",
            json={"name": "Updated"},
        )
        assert resp.status_code == 200
        assert resp.json()["name"] == "Updated"

    async def test_update_probe_set_pairs(self, client: AsyncClient):
        create_resp = await client.post(
            "/api/probe-sets",
            json={
                "name": "Pair Update",
                "pairs": [
                    {"pair_id": "p1", "prompt": "Q?", "completion": "A.", "role": "both"}
                ],
            },
        )
        probe_set_id = create_resp.json()["id"]

        new_pairs = [
            {"pair_id": "p1", "prompt": "Q1?", "completion": "A1.", "role": "train"},
            {"pair_id": "p2", "prompt": "Q2?", "completion": "A2.", "role": "query"},
        ]
        resp = await client.put(
            f"/api/probe-sets/{probe_set_id}",
            json={"pairs": new_pairs},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["pair_count"] == 2
        assert len(data["pairs"]) == 2

    async def test_update_probe_set_not_found(self, client: AsyncClient):
        resp = await client.put(
            "/api/probe-sets/nonexistent",
            json={"name": "Nope"},
        )
        assert resp.status_code == 404

    async def test_delete_probe_set(self, client: AsyncClient):
        create_resp = await client.post(
            "/api/probe-sets", json={"name": "To Delete"}
        )
        probe_set_id = create_resp.json()["id"]

        resp = await client.delete(f"/api/probe-sets/{probe_set_id}")
        assert resp.status_code == 204

        # Verify it's gone
        resp = await client.get(f"/api/probe-sets/{probe_set_id}")
        assert resp.status_code == 404

    async def test_delete_probe_set_not_found(self, client: AsyncClient):
        resp = await client.delete("/api/probe-sets/nonexistent")
        assert resp.status_code == 404

    async def test_create_empty_probe_set(self, client: AsyncClient):
        resp = await client.post(
            "/api/probe-sets", json={"name": "Empty Set"}
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["pair_count"] == 0
        assert data["pairs"] == []

    async def test_full_lifecycle(self, client: AsyncClient):
        """Full CRUD lifecycle test."""
        # Create
        resp = await client.post(
            "/api/probe-sets",
            json={
                "name": "Lifecycle",
                "pairs": [
                    {"pair_id": "p1", "prompt": "Q?", "completion": "A.", "role": "both"}
                ],
            },
        )
        assert resp.status_code == 201
        probe_set_id = resp.json()["id"]

        # Read
        resp = await client.get(f"/api/probe-sets/{probe_set_id}")
        assert resp.status_code == 200
        assert resp.json()["name"] == "Lifecycle"

        # Update
        resp = await client.put(
            f"/api/probe-sets/{probe_set_id}",
            json={"name": "Lifecycle Updated"},
        )
        assert resp.status_code == 200
        assert resp.json()["name"] == "Lifecycle Updated"

        # List
        resp = await client.get("/api/probe-sets")
        assert resp.status_code == 200
        assert len(resp.json()) == 1

        # Delete
        resp = await client.delete(f"/api/probe-sets/{probe_set_id}")
        assert resp.status_code == 204

        # Verify gone
        resp = await client.get("/api/probe-sets")
        assert resp.json() == []
