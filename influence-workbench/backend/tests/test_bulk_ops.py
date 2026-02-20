"""Integration tests for import/duplicate/bulk-role endpoints."""

from __future__ import annotations

import io
import json

import pytest


async def _create_probe_set(client, name="Test Set", pairs=None):
    """Helper to create a probe set and return its id."""
    if pairs is None:
        pairs = [
            {
                "pair_id": "p1",
                "prompt": "Hello",
                "completion": "World",
                "role": "both",
                "metadata": {},
            }
        ]
    resp = await client.post(
        "/api/probe-sets",
        json={"name": name, "pairs": pairs},
    )
    assert resp.status_code == 201
    return resp.json()["id"]


async def test_import_pairs_json(client):
    ps_id = await _create_probe_set(client)

    import_data = [
        {"pair_id": "imp1", "prompt": "A", "completion": "B", "role": "train"},
        {"pair_id": "imp2", "prompt": "C", "completion": "D", "role": "query"},
    ]
    content = json.dumps(import_data).encode()

    resp = await client.post(
        f"/api/probe-sets/{ps_id}/import-pairs",
        files={"file": ("pairs.json", content, "application/json")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["pair_count"] == 3  # 1 existing + 2 imported
    pair_ids = [p["pair_id"] for p in data["pairs"]]
    assert "p1" in pair_ids
    assert "imp1" in pair_ids
    assert "imp2" in pair_ids


async def test_import_pairs_csv(client):
    ps_id = await _create_probe_set(client)

    csv_content = "pair_id,prompt,completion,role\ncsv1,Hello,World,train\ncsv2,Foo,Bar,both\n"

    resp = await client.post(
        f"/api/probe-sets/{ps_id}/import-pairs",
        files={"file": ("pairs.csv", csv_content.encode(), "text/csv")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["pair_count"] == 3  # 1 existing + 2 imported


async def test_import_pairs_invalid_json(client):
    ps_id = await _create_probe_set(client)

    resp = await client.post(
        f"/api/probe-sets/{ps_id}/import-pairs",
        files={"file": ("bad.json", b"not json", "application/json")},
    )
    assert resp.status_code == 400


async def test_import_pairs_not_found(client):
    resp = await client.post(
        "/api/probe-sets/nonexistent/import-pairs",
        files={"file": ("pairs.json", b"[]", "application/json")},
    )
    assert resp.status_code == 404


async def test_bulk_role(client):
    pairs = [
        {"pair_id": "a", "prompt": "P1", "completion": "C1", "role": "both", "metadata": {}},
        {"pair_id": "b", "prompt": "P2", "completion": "C2", "role": "both", "metadata": {}},
        {"pair_id": "c", "prompt": "P3", "completion": "C3", "role": "both", "metadata": {}},
    ]
    ps_id = await _create_probe_set(client, pairs=pairs)

    resp = await client.post(
        f"/api/probe-sets/{ps_id}/bulk-role",
        json={"pair_ids": ["a", "c"], "role": "train"},
    )
    assert resp.status_code == 200
    data = resp.json()
    roles_by_id = {p["pair_id"]: p["role"] for p in data["pairs"]}
    assert roles_by_id["a"] == "train"
    assert roles_by_id["b"] == "both"  # unchanged
    assert roles_by_id["c"] == "train"


async def test_bulk_role_not_found(client):
    resp = await client.post(
        "/api/probe-sets/nonexistent/bulk-role",
        json={"pair_ids": ["a"], "role": "train"},
    )
    assert resp.status_code == 404
