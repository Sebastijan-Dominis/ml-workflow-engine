"""Simple end-to-end smoke tests for the FastAPI backend.

This test covers a short happy-path across multiple routers: health check,
file viewing and scripts execution (scripts execution is stubbed).
"""

from pathlib import Path
from typing import Any

import ml_service.backend.routers.scripts as scripts_router
import yaml


def test_backend_end_to_end(tmp_path: Path, monkeypatch: Any, fastapi_client: Any) -> None:
    # health
    resp = fastapi_client.get("/")
    assert resp.status_code == 200
    assert resp.json() == {"Healthy": 200}

    # file viewer
    f = tmp_path / "info.yaml"
    payload = {"hello": "world"}
    f.write_text(yaml.safe_dump(payload, sort_keys=False))

    resp = fastapi_client.post("/file_viewer/load", json={"path": str(f)})
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "yaml"
    parsed = yaml.safe_load(body["content"])
    assert parsed == payload

    # scripts: stub the execution helper to avoid starting subprocesses
    def fake_execute(module_path: str, payload, boolean_args=None):
        return {"exit_code": 0, "status": "SUCCESS", "stdout": "ok", "stderr": ""}

    monkeypatch.setattr(scripts_router, "execute_script", fake_execute)

    resp = fastapi_client.post("/scripts/generate_operator_hash", json={"operators": ["x"]})
    assert resp.status_code == 200
    assert resp.json()["exit_code"] == 0
