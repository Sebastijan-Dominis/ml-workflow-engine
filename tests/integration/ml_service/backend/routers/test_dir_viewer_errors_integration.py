"""Tests for error conditions in the `dir_viewer` router."""

from typing import Any


def test_dir_viewer_missing_path(fastapi_client: Any) -> None:
    resp = fastapi_client.post("/dir_viewer/load", json={})
    assert resp.status_code == 400


def test_dir_viewer_outside_repo(fastapi_client: Any) -> None:
    # Use a relative parent path that resolves outside the repo root
    resp = fastapi_client.post("/dir_viewer/load", json={"path": ".."})
    assert resp.status_code == 403


def test_dir_viewer_nonexistent_dir(fastapi_client: Any) -> None:
    resp = fastapi_client.post("/dir_viewer/load", json={"path": "tests/nonexistent_dir"})
    assert resp.status_code == 404
