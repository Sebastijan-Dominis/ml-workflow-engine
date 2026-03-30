"""Integration tests for ML service backend pipeline routes.

These tests use FastAPI's TestClient to exercise route validation and ensure
that the router forwards valid payloads to the underlying pipeline executor.
The actual subprocess invocation is monkeypatched to keep tests fast and
deterministic.
"""

from __future__ import annotations

from typing import Any

import ml_service.backend.routers.pipelines as pipelines_router_module
from fastapi.testclient import TestClient
from ml_service.backend.main import app


def test_pipelines_train_route_calls_execute_pipeline(monkeypatch: Any) -> None:
    """POST /pipelines/train should validate input and call execute_pipeline.

    The real subprocess call is replaced with a fake implementation that
    verifies the module path and payload shape. This focuses the test on the
    FastAPI routing, request validation and integration with the router.
    """

    fake_response: dict[str, Any] = {"exit_code": 0, "status": "SUCCESS", "stdout": "ok", "stderr": ""}

    def fake_execute_pipeline(module_path: str, payload: Any, boolean_args: list[str] | None = None) -> dict[str, Any]:
        # router for /train should execute the training runner module
        assert module_path == "pipelines.runners.train"
        # payload is a Pydantic model instance; check basic attributes
        assert hasattr(payload, "problem") and hasattr(payload, "segment") and hasattr(payload, "version")
        return fake_response

    monkeypatch.setattr(pipelines_router_module, "execute_pipeline", fake_execute_pipeline)

    client = TestClient(app)
    payload = {"problem": "no_show", "segment": "global", "version": "v1"}
    resp = client.post("/pipelines/train", json=payload)

    assert resp.status_code == 200
    assert resp.json() == fake_response
