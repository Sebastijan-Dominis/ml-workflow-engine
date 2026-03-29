import pytest


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
