"""Tests for error conditions in the `file_viewer` router."""

from pathlib import Path
from typing import Any


def test_file_viewer_missing_path(fastapi_client: Any) -> None:
    resp = fastapi_client.post("/file_viewer/load", json={})
    assert resp.status_code == 400


def test_file_viewer_nonexistent_file(tmp_path: Path, fastapi_client: Any) -> None:
    missing = tmp_path / "does_not_exist.yaml"
    resp = fastapi_client.post("/file_viewer/load", json={"path": str(missing)})
    assert resp.status_code == 404


def test_file_viewer_unsupported_type(tmp_path: Path, fastapi_client: Any) -> None:
    f = tmp_path / "notes.txt"
    f.write_text("hello")
    resp = fastapi_client.post("/file_viewer/load", json={"path": str(f)})
    assert resp.status_code == 400
