"""Integration tests for the `pipelines` FastAPI router."""

from typing import Any

import ml_service.backend.routers.pipelines as pipelines_router


def test_pipelines_train_search_and_run_all(monkeypatch: Any, fastapi_client: Any) -> None:
    called: dict[str, Any] = {"calls": []}

    def fake_execute_pipeline(module_path: str, payload, boolean_args=None):
        called["calls"].append({"module_path": module_path, "payload": getattr(payload, "model_dump", lambda **k: dict(payload))()})
        return {"exit_code": 0, "status": "SUCCESS", "stdout": "", "stderr": ""}

    monkeypatch.setattr(pipelines_router, "execute_pipeline", fake_execute_pipeline)

    # Train endpoint
    train_payload = {"problem": "cancellation", "segment": "all", "version": "v1"}
    resp = fastapi_client.post("/pipelines/train", json=train_payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["exit_code"] == 0
    assert any(call["module_path"] == "pipelines.runners.train" for call in called["calls"])

    # Search endpoint
    search_payload = {"problem": "cancellation", "segment": "all", "version": "v1"}
    resp = fastapi_client.post("/pipelines/search", json=search_payload)
    assert resp.status_code == 200
    assert any(call["module_path"] == "pipelines.search.search" for call in called["calls"])

    # Run all workflows uses defaults so empty payload is acceptable
    resp = fastapi_client.post("/pipelines/run_all_workflows", json={})
    assert resp.status_code == 200
    assert any(call["module_path"] == "pipelines.orchestration.master.run_all_workflows" for call in called["calls"])
"""Integration tests for the `pipelines` FastAPI router."""
