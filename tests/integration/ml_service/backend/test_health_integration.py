"""Integration tests for the ml_service backend health endpoint."""

from __future__ import annotations

from typing import Any


def test_health_check_root_endpoint(fastapi_client: Any) -> None:
    resp = fastapi_client.get("/")
    assert resp.status_code == 200
    assert resp.json() == {"Healthy": 200}
