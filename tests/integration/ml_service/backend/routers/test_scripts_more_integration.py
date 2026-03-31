"""Additional integration tests for the `scripts` router endpoints."""

from __future__ import annotations

from typing import Any

import ml_service.backend.routers.scripts as scripts_router


def test_generate_snapshot_binding_calls_execute_script(monkeypatch: Any, fastapi_client: Any) -> None:
    called: dict[str, Any] = {}

    def fake_execute_script(module_path: str, payload, boolean_args=None):
        called["module_path"] = module_path
        called["payload"] = getattr(payload, "model_dump", lambda **k: dict(payload))()
        return {"exit_code": 0, "status": "SUCCESS", "stdout": "", "stderr": ""}

    monkeypatch.setattr(scripts_router, "execute_script", fake_execute_script)

    resp = fastapi_client.post("/scripts/generate_snapshot_binding", json={"snapshot": "s"})
    assert resp.status_code == 200
    assert called.get("module_path") == "scripts.generators.generate_snapshot_binding"
    assert isinstance(called.get("payload"), dict)


def test_generate_cols_for_row_id_fingerprint_calls_execute_script(monkeypatch: Any, fastapi_client: Any) -> None:
    called: dict[str, Any] = {}

    def fake_execute_script(module_path: str, payload, boolean_args=None):
        called["module_path"] = module_path
        called["payload"] = getattr(payload, "model_dump", lambda **k: dict(payload))()
        return {"exit_code": 0, "status": "SUCCESS", "stdout": "", "stderr": ""}

    monkeypatch.setattr(scripts_router, "execute_script", fake_execute_script)

    resp = fastapi_client.post("/scripts/generate_cols_for_row_id_fingerprint", json={"col": "id"})
    assert resp.status_code == 200
    assert called.get("module_path") == "scripts.generators.generate_cols_for_row_id_fingerprint"
    assert isinstance(called.get("payload"), dict)
