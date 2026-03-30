"""Integration tests for the `features` FastAPI router."""

from pathlib import Path
from typing import Any

import ml_service.backend.routers.features as features_router


def test_features_validate_and_write(monkeypatch: Any, fastapi_client: Any, tmp_path: Path) -> None:
    # prepare fakes
    def fake_load_yaml_and_add_lineage(yaml_text: str) -> dict:
        return {"some": "value"}

    class Validated:
        def model_dump(self, mode: str = "json", **_: Any) -> dict:
            return {"validated": True}

    def fake_validate_feature_config(d: dict) -> Validated:
        return Validated()

    registry_dir = tmp_path / "registry"

    def fake_get_registry_path(root: Path) -> Path:
        return registry_dir

    def fake_registry_entry_exists(name: str, version: str, registry_path: Path) -> bool:
        return False

    called: dict[str, Any] = {}

    def fake_save_feature_registry(name: str, version: str, validated_config, registry_path: Path) -> dict:
        called["name"] = name
        called["version"] = version
        called["registry_path"] = registry_path
        return {"status": "saved", "path": str(registry_path / name / version)}

    monkeypatch.setattr(features_router, "load_yaml_and_add_lineage", fake_load_yaml_and_add_lineage)
    monkeypatch.setattr(features_router, "validate_feature_config", fake_validate_feature_config)
    monkeypatch.setattr(features_router, "get_registry_path", fake_get_registry_path)
    monkeypatch.setattr(features_router, "registry_entry_exists", fake_registry_entry_exists)
    monkeypatch.setattr(features_router, "save_feature_registry", fake_save_feature_registry)

    payload = {"name": "feat", "version": "v1", "config": "dummy: v1\n"}

    # validate
    resp = fastapi_client.post("/features/validate", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["valid"] is True
    assert body["exists"] is False
    assert "normalized" in body

    # write
    resp = fastapi_client.post("/features/write", json=payload)
    assert resp.status_code == 201
    body = resp.json()
    assert body.get("status") == "saved"
    assert called.get("name") == "feat"
