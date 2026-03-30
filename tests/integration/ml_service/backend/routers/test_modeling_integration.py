"""Integration tests for the `modeling` FastAPI router."""

from pathlib import Path
from typing import Any

import ml_service.backend.routers.modeling as modeling_router


def test_modeling_validate_and_write(monkeypatch: Any, fastapi_client: Any, tmp_path: Path) -> None:
    # Fake validated configs object with expected attributes
    class Small:
        def __init__(self, payload: dict):
            self._payload = payload

        def model_dump(self, mode: str = "json", exclude: set | None = None, **_: Any) -> dict:
            # ignore exclude for tests
            return dict(self._payload)

    class ValidatedConfigs:
        def __init__(self) -> None:
            self.model_specs = Small({"spec": 1})
            self.search = Small({"search": True})
            self.training = Small({"training": True})

    def fake_load_all_yamls_and_add_lineage(payload: dict) -> dict:
        return payload

    def fake_validate_all_configs(payload: dict) -> ValidatedConfigs:
        return ValidatedConfigs()

    def fake_check_paths(validated_configs: ValidatedConfigs) -> None:
        # validate endpoint expects this to run without error
        return None

    monkeypatch.setattr(modeling_router, "load_all_yamls_and_add_lineage", fake_load_all_yamls_and_add_lineage)
    monkeypatch.setattr(modeling_router, "validate_all_configs", fake_validate_all_configs)
    monkeypatch.setattr(modeling_router, "check_paths", fake_check_paths)

    payload = {"model_specs": "x", "search": "y", "training": "z"}

    resp = fastapi_client.post("/modeling/validate", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("valid") is True
    assert "normalized" in body

    # Write flow: check_paths returns an object with paths
    class Paths:
        def __init__(self) -> None:
            self.model_specs = "msp"
            self.search = "spp"
            self.training = "trp"

    def fake_check_paths_write(validated_configs: ValidatedConfigs) -> Paths:
        return Paths()

    called: dict[str, Any] = {}

    def fake_save_all_configs(validated_configs: ValidatedConfigs, paths: Paths) -> None:
        called["paths"] = paths

    monkeypatch.setattr(modeling_router, "check_paths", fake_check_paths_write)
    monkeypatch.setattr(modeling_router, "save_all_configs", fake_save_all_configs)

    resp = fastapi_client.post("/modeling/write", json=payload)
    assert resp.status_code == 201
    body = resp.json()
    assert "paths" in body
    assert body["paths"]["model_specs"] == "msp"
