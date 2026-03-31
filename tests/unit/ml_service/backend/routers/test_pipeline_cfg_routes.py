"""Tests for `ml_service.backend.routers.pipeline_cfg` validate and write branches."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest


def test_validate_pipeline_missing_config_payload() -> None:
    mod = importlib.import_module("ml_service.backend.routers.pipeline_cfg")
    orig = getattr(mod.validate_yaml, "__wrapped__", mod.validate_yaml)
    from fastapi import Request

    req = Request({"type": "http", "method": "POST", "path": "/", "headers": []})
    body = orig({}, req)
    assert body["valid"] is False
    assert "Missing config payload" in body["error"]


def test_validate_pipeline_missing_required_fields(monkeypatch) -> None:
    mod = importlib.import_module("ml_service.backend.routers.pipeline_cfg")
    # create a yaml with version but missing data_type/algorithm
    monkeypatch.setattr(mod, "load_yaml_and_add_lineage", lambda txt: {"version": "v1"})
    monkeypatch.setattr(mod, "validate_config_payload", lambda d: SimpleNamespace(model_dump=lambda mode="json": {}))

    orig = getattr(mod.validate_yaml, "__wrapped__", mod.validate_yaml)
    from fastapi import Request

    req = Request({"type": "http", "method": "POST", "path": "/", "headers": []})
    body = orig({"config": "version: v1"}, req)
    assert body["valid"] is False
    assert "Missing required fields" in body["error"]


def test_validate_pipeline_exists(monkeypatch) -> None:
    mod = importlib.import_module("ml_service.backend.routers.pipeline_cfg")
    monkeypatch.setattr(mod, "load_yaml_and_add_lineage", lambda txt: {"version": "v1"})
    monkeypatch.setattr(mod, "validate_config_payload", lambda d: SimpleNamespace(model_dump=lambda mode="json": {}))
    monkeypatch.setattr(mod, "get_config_path", lambda **k: SimpleNamespace(exists=lambda : True) )

    orig = getattr(mod.validate_yaml, "__wrapped__", mod.validate_yaml)
    from fastapi import Request
    payload = {"config": "version: v1", "data_type": "dt", "algorithm": "alg"}
    req = Request({"type": "http", "method": "POST", "path": "/", "headers": []})
    body = orig(payload, req)
    assert body["valid"] is True
    assert body["exists"] is True


def test_write_pipeline_exists_and_written(monkeypatch) -> None:
    mod = importlib.import_module("ml_service.backend.routers.pipeline_cfg")
    monkeypatch.setattr(mod, "load_yaml_and_add_lineage", lambda txt: {"version": "v1"})
    monkeypatch.setattr(mod, "validate_config_payload", lambda d: SimpleNamespace())

    # exists case
    monkeypatch.setattr(mod, "get_config_path", lambda **k: SimpleNamespace(exists=lambda : True))
    orig_write = getattr(mod.write_yaml, "__wrapped__", mod.write_yaml)
    from fastapi import Request
    payload = {"config": "version: v1", "data_type": "dt", "algorithm": "alg"}
    req = Request({"type": "http", "method": "POST", "path": "/", "headers": []})
    r = orig_write(payload, req)
    assert r["status"] == "exists"

    # written case
    fake_saved = {}
    def fake_save(config, config_path):
        fake_saved["ok"] = True

    monkeypatch.setattr(mod, "get_config_path", lambda **k: SimpleNamespace(exists=lambda : False))
    monkeypatch.setattr(mod, "save_config", fake_save)
    r2 = orig_write(payload, req)
    assert r2["success"] == "written"
    assert "path" in r2


def _fake_path(exists: bool, path_str: str = "/fake/pipeline/path"):
    class FakePath:
        def __init__(self, exists_val: bool):
            self._exists = exists_val

        def exists(self):
            return self._exists

        def __str__(self):
            return path_str

    return FakePath(exists)


def test_validate_yaml_success(monkeypatch):
    payload = {"config": "cfg" , "data_type": "tabular", "algorithm": "alg"}

    data_dict = {"version": "v1"}

    monkeypatch.setattr(
        "ml_service.backend.routers.pipeline_cfg.load_yaml_and_add_lineage",
        lambda text: data_dict,
    )

    class FakeValidated:
        def model_dump(self, mode="json"):
            return {"ok": True}

    monkeypatch.setattr(
        "ml_service.backend.routers.pipeline_cfg.validate_config_payload",
        lambda d: FakeValidated(),
    )

    monkeypatch.setattr(
        "ml_service.backend.routers.pipeline_cfg.get_config_path",
        lambda repo_root, data_type, algorithm, pipeline_version: _fake_path(True),
    )

    import ml_service.backend.routers.pipeline_cfg as pc_mod
    from fastapi import Request

    orig = getattr(pc_mod.validate_yaml, "__wrapped__", pc_mod.validate_yaml)
    req = Request({"type": "http", "method": "POST", "path": "/", "headers": []})
    j = orig(payload, req)

    assert j["valid"] is True
    assert j["exists"] is True
    assert j["normalized"]["ok"] is True


def test_validate_yaml_missing_fields(monkeypatch):
    # load returns no version -> function should return valid False
    monkeypatch.setattr(
        "ml_service.backend.routers.pipeline_cfg.load_yaml_and_add_lineage",
        lambda text: {},
    )
    # ensure payload validation does not raise so the router reaches the missing-fields check
    class _FakeVal:
        def model_dump(self, mode="json"):
            return {}

    monkeypatch.setattr(
        "ml_service.backend.routers.pipeline_cfg.validate_config_payload",
        lambda d: _FakeVal(),
    )

    import ml_service.backend.routers.pipeline_cfg as pc_mod
    from fastapi import Request

    orig = getattr(pc_mod.validate_yaml, "__wrapped__", pc_mod.validate_yaml)
    req = Request({"type": "http", "method": "POST", "path": "/", "headers": []})
    j = orig({"config": "x"}, req)
    assert j["valid"] is False
    assert "Missing required fields" in j["error"]


def test_write_yaml_exists(monkeypatch):
    payload = {"config": "cfg", "data_type": "tabular", "algorithm": "alg"}
    data_dict = {"version": "v2"}

    monkeypatch.setattr(
        "ml_service.backend.routers.pipeline_cfg.load_yaml_and_add_lineage",
        lambda text: data_dict,
    )
    monkeypatch.setattr(
        "ml_service.backend.routers.pipeline_cfg.validate_config_payload",
        lambda d: True,
    )
    monkeypatch.setattr(
        "ml_service.backend.routers.pipeline_cfg.get_config_path",
        lambda repo_root, data_type, algorithm, pipeline_version: _fake_path(True),
    )

    import ml_service.backend.routers.pipeline_cfg as pc_mod
    from fastapi import Request

    orig = getattr(pc_mod.write_yaml, "__wrapped__", pc_mod.write_yaml)
    req = Request({"type": "http", "method": "POST", "path": "/", "headers": []})
    j = orig(payload, req)
    assert j["status"] == "exists"


def test_write_yaml_written_and_save_called(monkeypatch, tmp_path):
    payload = {"config": "cfg", "data_type": "tabular", "algorithm": "alg"}
    data_dict = {"version": "v3"}

    monkeypatch.setattr(
        "ml_service.backend.routers.pipeline_cfg.load_yaml_and_add_lineage",
        lambda text: data_dict,
    )
    monkeypatch.setattr(
        "ml_service.backend.routers.pipeline_cfg.validate_config_payload",
        lambda d: True,
    )

    fake_path = _fake_path(False, "/pipeline/written")
    monkeypatch.setattr(
        "ml_service.backend.routers.pipeline_cfg.get_config_path",
        lambda repo_root, data_type, algorithm, pipeline_version: fake_path,
    )

    called = {}

    def _save_config(config, config_path):
        called["c"] = (config, config_path)

    monkeypatch.setattr("ml_service.backend.routers.pipeline_cfg.save_config", _save_config)

    import ml_service.backend.routers.pipeline_cfg as pc_mod
    from fastapi import Request

    orig = getattr(pc_mod.write_yaml, "__wrapped__", pc_mod.write_yaml)
    req = Request({"type": "http", "method": "POST", "path": "/", "headers": []})
    j = orig(payload, req)

    assert j["success"] == "written"
    assert j["path"] == str(fake_path)
    assert "c" in called


def test_write_yaml_save_failure_raises(monkeypatch):
    payload = {"config": "cfg", "data_type": "tabular", "algorithm": "alg"}
    data_dict = {"version": "v4"}

    monkeypatch.setattr(
        "ml_service.backend.routers.pipeline_cfg.load_yaml_and_add_lineage",
        lambda text: data_dict,
    )
    monkeypatch.setattr(
        "ml_service.backend.routers.pipeline_cfg.validate_config_payload",
        lambda d: True,
    )

    fake_path = _fake_path(False, "/will/fail")
    monkeypatch.setattr(
        "ml_service.backend.routers.pipeline_cfg.get_config_path",
        lambda repo_root, data_type, algorithm, pipeline_version: fake_path,
    )

    def _bad_save(config, config_path):
        raise RuntimeError("no space")

    monkeypatch.setattr("ml_service.backend.routers.pipeline_cfg.save_config", _bad_save)

    import ml_service.backend.routers.pipeline_cfg as pc_mod
    from fastapi import Request

    orig = getattr(pc_mod.write_yaml, "__wrapped__", pc_mod.write_yaml)
    req = Request({"type": "http", "method": "POST", "path": "/", "headers": []})
    with pytest.raises(Exception) as exc:
        orig(payload, req)

    assert "no space" in str(exc.value)


def test_write_yaml_missing_config_payload_raises() -> None:
    mod = importlib.import_module("ml_service.backend.routers.pipeline_cfg")
    orig = getattr(mod.write_yaml, "__wrapped__", mod.write_yaml)
    from fastapi import Request

    req = Request({"type": "http", "method": "POST", "path": "/", "headers": []})
    with pytest.raises(Exception) as exc:
        orig({}, req)
    assert "Missing config payload" in str(exc.value)


def test_write_yaml_missing_required_fields_raises(monkeypatch) -> None:
    mod = importlib.import_module("ml_service.backend.routers.pipeline_cfg")
    monkeypatch.setattr(mod, "load_yaml_and_add_lineage", lambda txt: {"version": "v1"})
    monkeypatch.setattr(mod, "validate_config_payload", lambda d: True)

    orig = getattr(mod.write_yaml, "__wrapped__", mod.write_yaml)
    from fastapi import Request

    req = Request({"type": "http", "method": "POST", "path": "/", "headers": []})
    with pytest.raises(Exception) as exc:
        orig({"config": "version: v1"}, req)
    assert "Missing required fields" in str(exc.value)
