import importlib

import pytest
from fastapi import Request


class DummyModel:
    def __init__(self, payload):
        self._payload = payload

    def model_dump(self, mode=None):
        return {"dumped": True, "payload": self._payload}


def test_validate_yaml_missing_fields_returns_invalid():
    mod = importlib.import_module("ml_service.backend.routers.features")
    orig = getattr(mod.validate_yaml, "__wrapped__", mod.validate_yaml)
    req = Request({"type": "http", "method": "POST", "path": "/", "headers": []})

    res = orig({}, req)
    assert res["valid"] is False


def test_validate_yaml_success_and_exists(monkeypatch, tmp_path):
    mod = importlib.import_module("ml_service.backend.routers.features")
    orig = getattr(mod.validate_yaml, "__wrapped__", mod.validate_yaml)

    # stub dependencies
    monkeypatch.setattr(mod, "load_yaml_and_add_lineage", lambda txt: {"k": "v"})
    monkeypatch.setattr(mod, "validate_feature_config", lambda data: DummyModel(data))
    monkeypatch.setattr(mod, "get_registry_path", lambda p: tmp_path, raising=False)
    monkeypatch.setattr(mod, "registry_entry_exists", lambda name, version, p: True)

    req = Request({"type": "http", "method": "POST", "path": "/", "headers": []})
    payload = {"name": "n", "version": "v", "config": "yaml: 1"}
    res = orig(payload, req)

    assert res["valid"] is True
    assert res["exists"] is True
    assert "normalized" in res


def test_write_yaml_exists_and_write(monkeypatch, tmp_path):
    mod = importlib.import_module("ml_service.backend.routers.features")
    orig = getattr(mod.write_yaml, "__wrapped__", mod.write_yaml)

    monkeypatch.setattr(mod, "load_yaml_and_add_lineage", lambda txt: {"k": "v"})
    monkeypatch.setattr(mod, "validate_feature_config", lambda data: DummyModel(data))
    monkeypatch.setattr(mod, "get_registry_path", lambda p: tmp_path, raising=False)

    # case: exists -> short-circuit
    monkeypatch.setattr(mod, "registry_entry_exists", lambda n, v, p: True)
    payload = {"name": "n", "version": "v", "config": "yaml: 1"}
    res = orig(payload, Request({"type": "http", "method": "POST", "path": "/", "headers": []}))
    assert res["status"] == "exists"

    # case: save proceeds
    monkeypatch.setattr(mod, "registry_entry_exists", lambda n, v, p: False)
    monkeypatch.setattr(mod, "save_feature_registry", lambda name, version, validated_config, registry_path: {"status": "saved"})
    res2 = orig(payload, Request({"type": "http", "method": "POST", "path": "/", "headers": []}))
    assert res2 == {"status": "saved"}


def test_validate_yaml_success(monkeypatch):
    payload = {"name": "feat", "version": "v1", "config": "yaml: x"}

    data_with_lineage = {"a": 1}

    monkeypatch.setattr(
        "ml_service.backend.routers.features.load_yaml_and_add_lineage",
        lambda text: data_with_lineage,
    )

    class FakeValidated:
        def model_dump(self, mode="json"):
            return {"ok": True}

    monkeypatch.setattr(
        "ml_service.backend.routers.features.validate_feature_config",
        lambda d: FakeValidated(),
    )

    monkeypatch.setattr(
        "ml_service.backend.routers.features.get_registry_path",
        lambda repo_root: "/registry/path",
    )

    monkeypatch.setattr(
        "ml_service.backend.routers.features.registry_entry_exists",
        lambda name, version, path: True,
    )

    import ml_service.backend.routers.features as fmod
    from fastapi import Request

    orig = getattr(fmod.validate_yaml, "__wrapped__", fmod.validate_yaml)
    req = Request({"type": "http", "method": "POST", "path": "/", "headers": []})
    j = orig(payload, req)

    assert j["valid"] is True
    assert j["exists"] is True
    assert j["normalized"]["ok"] is True


def test_validate_yaml_missing_fields(monkeypatch):
    import ml_service.backend.routers.features as fmod
    from fastapi import Request

    orig = getattr(fmod.validate_yaml, "__wrapped__", fmod.validate_yaml)
    req = Request({"type": "http", "method": "POST", "path": "/", "headers": []})
    j = orig({"config": "x"}, req)
    assert j["valid"] is False
    assert "Missing feature set name or version" in j["error"]


def test_write_yaml_exists(monkeypatch):
    payload = {"name": "f", "version": "v2", "config": "y"}

    monkeypatch.setattr(
        "ml_service.backend.routers.features.load_yaml_and_add_lineage",
        lambda text: {"a": 1},
    )
    monkeypatch.setattr(
        "ml_service.backend.routers.features.validate_feature_config",
        lambda d: True,
    )
    monkeypatch.setattr(
        "ml_service.backend.routers.features.get_registry_path",
        lambda repo_root: "/registry/path",
    )
    monkeypatch.setattr(
        "ml_service.backend.routers.features.registry_entry_exists",
        lambda name, version, path: True,
    )

    import ml_service.backend.routers.features as fmod
    from fastapi import Request

    orig = getattr(fmod.write_yaml, "__wrapped__", fmod.write_yaml)
    req = Request({"type": "http", "method": "POST", "path": "/", "headers": []})
    j = orig(payload, req)
    assert j["status"] == "exists"


def test_write_yaml_saved_and_returned(monkeypatch):
    payload = {"name": "f", "version": "v3", "config": "y"}

    monkeypatch.setattr(
        "ml_service.backend.routers.features.load_yaml_and_add_lineage",
        lambda text: {"a": 1},
    )
    monkeypatch.setattr(
        "ml_service.backend.routers.features.validate_feature_config",
        lambda d: True,
    )
    monkeypatch.setattr(
        "ml_service.backend.routers.features.get_registry_path",
        lambda repo_root: "/registry/path",
    )
    monkeypatch.setattr(
        "ml_service.backend.routers.features.registry_entry_exists",
        lambda name, version, path: False,
    )

    def fake_save(name, version, validated_config, registry_path):
        return {"ok": True, "name": name, "version": version}

    monkeypatch.setattr(
        "ml_service.backend.routers.features.save_feature_registry",
        fake_save,
    )

    import ml_service.backend.routers.features as fmod
    from fastapi import Request

    orig = getattr(fmod.write_yaml, "__wrapped__", fmod.write_yaml)
    req = Request({"type": "http", "method": "POST", "path": "/", "headers": []})
    j = orig(payload, req)

    assert j["ok"] is True
    assert j["name"] == "f"


def test_write_yaml_save_failure_raises(monkeypatch):
    payload = {"name": "f", "version": "v4", "config": "y"}

    monkeypatch.setattr(
        "ml_service.backend.routers.features.load_yaml_and_add_lineage",
        lambda text: {"a": 1},
    )
    monkeypatch.setattr(
        "ml_service.backend.routers.features.validate_feature_config",
        lambda d: True,
    )
    monkeypatch.setattr(
        "ml_service.backend.routers.features.get_registry_path",
        lambda repo_root: "/registry/path",
    )
    monkeypatch.setattr(
        "ml_service.backend.routers.features.registry_entry_exists",
        lambda name, version, path: False,
    )

    def _bad_save(name, version, validated_config, registry_path):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "ml_service.backend.routers.features.save_feature_registry",
        _bad_save,
    )

    import ml_service.backend.routers.features as fmod
    from fastapi import Request

    orig = getattr(fmod.write_yaml, "__wrapped__", fmod.write_yaml)
    req = Request({"type": "http", "method": "POST", "path": "/", "headers": []})
    with pytest.raises(Exception) as exc:
        orig(payload, req)

    assert "boom" in str(exc.value)
