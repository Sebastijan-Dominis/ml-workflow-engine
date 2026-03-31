"""Integration tests for the `promotion_thresholds` FastAPI router."""

from pathlib import Path
from typing import Any

import ml_service.backend.routers.promotion_thresholds as pt_router


def test_promotion_thresholds_validate_and_write(monkeypatch: Any, fastapi_client: Any, tmp_path: Path) -> None:
    def fake_load_yaml_and_add_lineage(yaml_text: str) -> dict:
        return {"thresholds": []}

    class Validated:
        def model_dump(self, mode: str = "json", **_: Any) -> dict:
            return {"validated": True}

    def fake_validate_config_payload(d: dict) -> Validated:
        return Validated()

    # Simulate thresholds not existing yet
    def fake_check_thresholds_exist(config_path: Path, problem_type: str, segment: str) -> tuple[bool, dict]:
        return False, {}

    called: dict[str, Any] = {}

    def fake_save_promotion_thresholds(thresholds: dict, validated: Validated, config_path: Path, problem_type: str, segment: str) -> None:
        called["saved"] = True
        called["config_path"] = config_path

    monkeypatch.setattr(pt_router, "load_yaml_and_add_lineage", fake_load_yaml_and_add_lineage)
    monkeypatch.setattr(pt_router, "validate_config_payload", fake_validate_config_payload)
    monkeypatch.setattr(pt_router, "check_thresholds_exist", fake_check_thresholds_exist)
    monkeypatch.setattr(pt_router, "save_promotion_thresholds", fake_save_promotion_thresholds)

    payload = {"config": "dummy: v1\n", "problem_type": "cancellation", "segment": "all"}

    resp = fastapi_client.post("/promotion_thresholds/validate", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["valid"] is True
    assert body["exists"] is False

    resp = fastapi_client.post("/promotion_thresholds/write", json=payload)
    assert resp.status_code == 201
    body = resp.json()
    assert body.get("success") == "written"
    assert called.get("saved") is True
