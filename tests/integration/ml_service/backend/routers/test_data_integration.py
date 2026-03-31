"""Integration tests for the `data` FastAPI router."""

from pathlib import Path
from typing import Any

import ml_service.backend.routers.data as data_router


def test_data_validate_and_write(monkeypatch: Any, fastapi_client: Any, tmp_path: Path) -> None:
    # Fake YAML loader to return expected data keys
    def fake_load_yaml_and_add_lineage(yaml_text: str) -> dict:
        return {"data": {"name": "hotel_bookings", "version": "v1"}}

    def fake_validate_config_payload(config_type: str, data_dict: dict) -> None:
        # returns nothing but should not raise on valid input
        return None

    cfg_path = tmp_path / "configs" / "data" / "interim" / "hotel_bookings" / "v1"

    def fake_get_config_path(repo_root: str, config_type: str, dataset_name: str, dataset_version: str) -> Path:
        return cfg_path

    called: dict[str, Any] = {}

    def fake_save_config(config: dict, config_path: Path) -> None:
        called["config_path"] = config_path
        config_path.parent.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(data_router, "load_yaml_and_add_lineage", fake_load_yaml_and_add_lineage)
    monkeypatch.setattr(data_router, "validate_config_payload", fake_validate_config_payload)
    monkeypatch.setattr(data_router, "get_config_path", fake_get_config_path)
    monkeypatch.setattr(data_router, "save_config", fake_save_config)

    payload = {"type": "interim", "config": "data:\n  name: hotel_bookings\n  version: v1\n"}

    resp = fastapi_client.post("/data/validate", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["valid"] is True
    assert body["exists"] is False

    resp = fastapi_client.post("/data/write", json=payload)
    assert resp.status_code == 201
    body = resp.json()
    assert body.get("status") == "written"
    assert isinstance(called.get("config_path"), Path)
