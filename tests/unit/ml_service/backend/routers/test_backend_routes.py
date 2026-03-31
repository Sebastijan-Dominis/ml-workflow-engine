"""Integration-style tests for ml_service backend routers using TestClient.

These tests monkeypatch internal helpers to avoid heavy side-effects and
verify that endpoints route requests and return expected JSON shapes.
"""
from __future__ import annotations

from typing import Any


def test_pipelines_train_endpoint(fastapi_client, monkeypatch):
    import ml_service.backend.routers.pipelines as pipelines_router

    def fake_execute(module_path: str, payload: Any, boolean_args: list[str] | None = None):
        return {"executed": module_path, "payload": payload.model_dump() if hasattr(payload, "model_dump") else dict(payload)}

    monkeypatch.setattr(pipelines_router, "execute_pipeline", fake_execute)

    res = fastapi_client.post("/pipelines/train", json={"problem": "p", "segment": "s", "version": "v"})
    assert res.status_code == 200
    j = res.json()
    assert j["executed"] == "pipelines.runners.train"


def test_scripts_generate_fake_data(fastapi_client, monkeypatch):
    import ml_service.backend.routers.scripts as scripts_router

    def fake_exec(module_path: str, payload: Any, boolean_args: list[str] | None = None):
        return {"script": module_path, "args": getattr(payload, "model_dump", lambda **k: dict(payload))()}

    monkeypatch.setattr(scripts_router, "execute_script", fake_exec)

    res = fastapi_client.post("/scripts/generate_fake_data", json={"data": "hotel_bookings", "version": "v1"})
    assert res.status_code == 200
    assert res.json()["script"] == "scripts.generators.generate_fake_data"


def test_pipeline_cfg_validate_and_write(fastapi_client, monkeypatch, tmp_path):
    import ml_service.backend.routers.pipeline_cfg as pcfg

    # Stub loader + validator
    monkeypatch.setattr(pcfg, "load_yaml_and_add_lineage", lambda yaml_text: {"version": "v1"})

    class DummyValidated:
        def model_dump(self, mode: str = "json"):
            return {"normalized": True}

    monkeypatch.setattr(pcfg, "validate_config_payload", lambda data: DummyValidated())

    # Point get_config_path to a path under tmp_path that exists
    def fake_get_config_path(*, repo_root: str, data_type: str, algorithm: str, pipeline_version: str):
        p = tmp_path / data_type / algorithm
        p.mkdir(parents=True, exist_ok=True)
        fp = p / f"{pipeline_version}.yaml"
        fp.write_text("x: 1")
        return fp

    monkeypatch.setattr(pcfg, "get_config_path", fake_get_config_path)

    # Validate should report exists=True
    res = fastapi_client.post("/pipeline_cfg/validate", json={"config": "dummy", "data_type": "dt", "algorithm": "alg"})
    assert res.status_code == 200
    j = res.json()
    assert j["valid"] is True
    assert j["exists"] is True

    # Now test write: make get_config_path return non-existing file and patch save_config
    def fake_get_config_path2(*, repo_root: str, data_type: str, algorithm: str, pipeline_version: str):
        p = tmp_path / "new" / algorithm
        p.mkdir(parents=True, exist_ok=True)
        return p / f"{pipeline_version}.yaml"

    monkeypatch.setattr(pcfg, "get_config_path", fake_get_config_path2)

    monkeypatch.setattr(pcfg, "save_config", lambda config, config_path: None)

    res2 = fastapi_client.post("/pipeline_cfg/write", json={"config": "dummy", "data_type": "dt2", "algorithm": "alg2"})
    assert res2.status_code == 201
    j2 = res2.json()
    assert j2["success"] == "written"


def test_features_validate_and_write(fastapi_client, monkeypatch, tmp_path):
    import ml_service.backend.routers.features as features_router

    monkeypatch.setattr(features_router, "load_yaml_and_add_lineage", lambda yaml_text: {"some": "data"})

    class DummyVal:
        def model_dump(self, mode: str = "json"):
            return {"ok": True}

    monkeypatch.setattr(features_router, "validate_feature_config", lambda d: DummyVal())

    # Simulate registry path
    monkeypatch.setattr(features_router, "get_registry_path", lambda repo_root: tmp_path / "features.yaml")

    monkeypatch.setattr(features_router, "registry_entry_exists", lambda name, version, registry_path: False)
    monkeypatch.setattr(features_router, "save_feature_registry", lambda name, version, validated_config, registry_path: {"saved": True})

    res = fastapi_client.post("/features/validate", json={"name": "n", "version": "v", "config": "yaml"})
    assert res.status_code == 200
    assert res.json()["valid"] is True

    res2 = fastapi_client.post("/features/write", json={"name": "n2", "version": "v2", "config": "yaml"})
    assert res2.status_code == 200 or res2.status_code == 201


def test_file_viewer_and_dir_viewer_load(fastapi_client, tmp_path, monkeypatch):
    # File viewer: write a small YAML file and request it
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text("a: 1")

    res = fastapi_client.post("/file_viewer/load", json={"path": str(yaml_path)})
    assert res.status_code == 200
    j = res.json()
    assert "content" in j and "mode" in j

    # Dir viewer: set repo_root to tmp_path and create directory
    import ml_service.backend.routers.dir_viewer as dir_router

    monkeypatch.setattr(dir_router, "repo_root", str(tmp_path))
    (tmp_path / "some_dir").mkdir()
    (tmp_path / "some_dir" / "f.txt").write_text("x")

    res2 = fastapi_client.post("/dir_viewer/load", json={"path": "some_dir"})
    assert res2.status_code == 200
    j2 = res2.json()
    assert "tree" in j2 and "tree_yaml" in j2


def test_promotion_thresholds_validate_and_write(fastapi_client, monkeypatch, tmp_path):
    import ml_service.backend.routers.promotion_thresholds as prom_router

    monkeypatch.setattr(prom_router, "load_yaml_and_add_lineage", lambda yaml_text: {"x": 1})

    class DummyVal:
        def model_dump(self, mode: str = "json"):
            return {"ok": True}

    monkeypatch.setattr(prom_router, "validate_config_payload", lambda d: DummyVal())

    thresholds_path = tmp_path / "configs" / "promotion" / "thresholds.yaml"
    thresholds_path.parent.mkdir(parents=True, exist_ok=True)

    # Case: not exists
    monkeypatch.setattr(prom_router, "check_thresholds_exist", lambda config_path, problem_type, segment: (False, {}))
    monkeypatch.setattr(prom_router, "save_promotion_thresholds", lambda **kwargs: None)

    res = fastapi_client.post("/promotion_thresholds/validate", json={"config": "x", "problem_type": "p", "segment": "s"})
    assert res.status_code == 200
    assert res.json()["valid"] is True

    res2 = fastapi_client.post("/promotion_thresholds/write", json={"config": "x", "problem_type": "p", "segment": "s"})
    assert res2.status_code == 201
    assert res2.json()["success"] == "written"
