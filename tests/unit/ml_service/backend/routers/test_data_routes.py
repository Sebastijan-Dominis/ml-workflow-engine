import pytest


def _fake_path(exists: bool, path_str: str = "/fake/path"):
    class FakePath:
        def __init__(self, exists_val: bool):
            self._exists = exists_val

        def exists(self):
            return self._exists

        def __str__(self):
            return path_str

    return FakePath(exists)


def test_validate_yaml_success(monkeypatch, fastapi_client):
    payload = {"type": "interim", "config": "dummy: yaml"}

    data_dict = {"data": {"name": "ds", "version": "v1"}, "lineage": {"created_at": "t"}}

    monkeypatch.setattr(
        "ml_service.backend.routers.data.load_yaml_and_add_lineage",
        lambda text: data_dict,
    )
    monkeypatch.setattr(
        "ml_service.backend.routers.data.validate_config_payload",
        lambda config_type, d: True,
    )
    monkeypatch.setattr(
        "ml_service.backend.routers.data.get_config_path",
        lambda repo_root, config_type, dataset_name, dataset_version: _fake_path(True),
    )

    import ml_service.backend.routers.data as data_mod
    from fastapi import Request

    orig = getattr(data_mod.validate_yaml, "__wrapped__", data_mod.validate_yaml)
    req = Request({"type": "http", "method": "POST", "path": "/", "headers": []})
    body = orig(payload, req)
    assert body["valid"] is True
    assert body["exists"] is True
    assert body["normalized"] == data_dict


def test_validate_yaml_missing_fields(monkeypatch, fastapi_client):
    # missing type should produce valid=False with an error message
    import ml_service.backend.routers.data as data_mod
    from fastapi import Request

    orig = getattr(data_mod.validate_yaml, "__wrapped__", data_mod.validate_yaml)
    req = Request({"type": "http", "method": "POST", "path": "/", "headers": []})
    body = orig({"config": "x"}, req)
    assert body["valid"] is False
    assert "Missing or invalid config type" in body["error"]


def test_write_yaml_exists(monkeypatch, fastapi_client):
    payload = {"type": "processed", "config": "dummy: yaml"}

    data_dict = {"data": {"name": "ds2", "version": "v2"}, "lineage": {"created_at": "t"}}

    monkeypatch.setattr(
        "ml_service.backend.routers.data.load_yaml_and_add_lineage",
        lambda text: data_dict,
    )
    monkeypatch.setattr(
        "ml_service.backend.routers.data.validate_config_payload",
        lambda config_type, d: True,
    )
    monkeypatch.setattr(
        "ml_service.backend.routers.data.get_config_path",
        lambda repo_root, config_type, dataset_name, dataset_version: _fake_path(True, "/exists/path"),
    )

    import ml_service.backend.routers.data as data_mod
    from fastapi import Request

    orig = getattr(data_mod.write_yaml, "__wrapped__", data_mod.write_yaml)
    req = Request({"type": "http", "method": "POST", "path": "/", "headers": []})
    body = orig(payload, req)
    assert body["status"] == "exists"


def test_write_yaml_written_and_save_called(monkeypatch, fastapi_client):
    payload = {"type": "processed", "config": "dummy: yaml"}

    data_dict = {"data": {"name": "ds3", "version": "v3"}, "lineage": {"created_at": "t"}}

    monkeypatch.setattr(
        "ml_service.backend.routers.data.load_yaml_and_add_lineage",
        lambda text: data_dict,
    )
    monkeypatch.setattr(
        "ml_service.backend.routers.data.validate_config_payload",
        lambda config_type, d: True,
    )

    fake_path = _fake_path(False, "/written/path")
    monkeypatch.setattr(
        "ml_service.backend.routers.data.get_config_path",
        lambda repo_root, config_type, dataset_name, dataset_version: fake_path,
    )

    called = {}

    def _save_config(payload_dict, path):
        called["called_with"] = (payload_dict, path)

    monkeypatch.setattr("ml_service.backend.routers.data.save_config", _save_config)

    import ml_service.backend.routers.data as data_mod
    from fastapi import Request

    orig = getattr(data_mod.write_yaml, "__wrapped__", data_mod.write_yaml)
    req = Request({"type": "http", "method": "POST", "path": "/", "headers": []})
    body = orig(payload, req)
    assert body["status"] == "written"
    assert body["path"] == str(fake_path)
    assert "called_with" in called


def test_write_yaml_save_failure_raises(monkeypatch, fastapi_client):
    payload = {"type": "processed", "config": "dummy: yaml"}

    data_dict = {"data": {"name": "ds4", "version": "v4"}, "lineage": {"created_at": "t"}}

    monkeypatch.setattr(
        "ml_service.backend.routers.data.load_yaml_and_add_lineage",
        lambda text: data_dict,
    )
    monkeypatch.setattr(
        "ml_service.backend.routers.data.validate_config_payload",
        lambda config_type, d: True,
    )

    fake_path = _fake_path(False, "/will/fail")
    monkeypatch.setattr(
        "ml_service.backend.routers.data.get_config_path",
        lambda repo_root, config_type, dataset_name, dataset_version: fake_path,
    )

    def _save_config_fail(payload_dict, path):
        raise RuntimeError("disk write error")

    monkeypatch.setattr("ml_service.backend.routers.data.save_config", _save_config_fail)

    import ml_service.backend.routers.data as data_mod
    from fastapi import Request

    orig = getattr(data_mod.write_yaml, "__wrapped__", data_mod.write_yaml)
    req = Request({"type": "http", "method": "POST", "path": "/", "headers": []})
    with pytest.raises(Exception) as exc:
        orig(payload, req)

    # the router wraps failures in HTTPException with the original message
    assert "disk write error" in str(exc.value)
