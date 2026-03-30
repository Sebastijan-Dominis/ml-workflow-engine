"""Integration tests for the `pipeline_cfg` FastAPI router."""

from pathlib import Path
from typing import Any

import ml_service.backend.routers.pipeline_cfg as pcfg_router


def test_pipeline_cfg_validate_happy_path(monkeypatch: Any, fastapi_client: Any, tmp_path: Path) -> None:
    data_dict: dict[str, str] = {"version": "v1"}

    def fake_load_yaml_and_add_lineage(yaml_text: str) -> dict:
        return data_dict

    class Validated:
        def model_dump(self, mode: str = "json", **_: Any) -> dict:
            return {"version": data_dict["version"], "normalized": True}

    def fake_validate_config_payload(d: dict) -> Validated:
        return Validated()

    cfg_path = tmp_path / "cfgs" / "hotel" / "alg" / "v1"

    def fake_get_config_path(repo_root: str, data_type: str, algorithm: str, pipeline_version: str) -> Path:
        return cfg_path

    monkeypatch.setattr(pcfg_router, "load_yaml_and_add_lineage", fake_load_yaml_and_add_lineage)
    monkeypatch.setattr(pcfg_router, "validate_config_payload", fake_validate_config_payload)
    monkeypatch.setattr(pcfg_router, "get_config_path", fake_get_config_path)

    payload = {"config": "version: v1\n", "data_type": "hotel", "algorithm": "alg"}

    resp = fastapi_client.post("/pipeline_cfg/validate", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["valid"] is True
    assert body["exists"] is False
    assert "normalized" in body


def test_pipeline_cfg_write_success(monkeypatch: Any, fastapi_client: Any, tmp_path: Path) -> None:
    data_dict: dict[str, str] = {"version": "v1"}

    def fake_load_yaml_and_add_lineage(yaml_text: str) -> dict:
        return data_dict

    class Validated:
        def model_dump(self, mode: str = "json", **_: Any) -> dict:
            return {"version": data_dict["version"], "normalized": True}

    def fake_validate_config_payload(d: dict) -> Validated:
        return Validated()

    cfg_path = tmp_path / "cfgs" / "hotel" / "alg" / "v1"

    def fake_get_config_path(repo_root: str, data_type: str, algorithm: str, pipeline_version: str) -> Path:
        return cfg_path

    called: dict[str, Any] = {}

    def fake_save_config(config: dict, config_path: Path) -> None:
        called["config"] = config
        called["config_path"] = config_path
        config_path.parent.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(pcfg_router, "load_yaml_and_add_lineage", fake_load_yaml_and_add_lineage)
    monkeypatch.setattr(pcfg_router, "validate_config_payload", fake_validate_config_payload)
    monkeypatch.setattr(pcfg_router, "get_config_path", fake_get_config_path)
    monkeypatch.setattr(pcfg_router, "save_config", fake_save_config)

    payload = {"config": "version: v1\n", "data_type": "hotel", "algorithm": "alg"}

    resp = fastapi_client.post("/pipeline_cfg/write", json=payload)
    assert resp.status_code == 201
    body = resp.json()
    assert body.get("success") == "written"
    assert "path" in body
