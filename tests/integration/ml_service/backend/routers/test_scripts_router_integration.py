"""Integration tests for the `scripts` FastAPI router.

These tests stub the underlying `execute_script` helper so the HTTP route
can be exercised without launching subprocesses.
"""

from typing import Any

import ml_service.backend.routers.scripts as scripts_router


def test_generate_operator_hash_route(monkeypatch: Any, fastapi_client: Any) -> None:
    called: dict[str, Any] = {}

    def fake_execute_script(module_path: str, payload, boolean_args=None):
        called["module_path"] = module_path
        # payload is a pydantic model converted to dict by FastAPI
        called["payload"] = getattr(payload, "model_dump", lambda **k: dict(payload))()
        return {"exit_code": 0, "status": "SUCCESS", "stdout": "ok", "stderr": ""}

    monkeypatch.setattr(scripts_router, "execute_script", fake_execute_script)

    resp = fastapi_client.post("/scripts/generate_operator_hash", json={"operators": ["opA", "opB"]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["exit_code"] == 0
    assert called.get("module_path") == "scripts.generators.generate_operator_hash"
    payload = called.get("payload")
    assert isinstance(payload, dict)
    assert payload.get("operators") == ["opA", "opB"]
