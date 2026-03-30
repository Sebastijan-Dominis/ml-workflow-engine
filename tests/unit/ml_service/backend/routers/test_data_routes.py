"""Tests for ``ml_service.backend.routers.data`` validation and write helpers.

These tests monkeypatch the module-local helpers to exercise the "exists"
and "written" branches as well as the missing/invalid input path.
"""

from __future__ import annotations

import importlib
from typing import Any

import pytest

mod = importlib.import_module("ml_service.backend.routers.data")


class FakePath:
    def __init__(self, exists_flag: bool) -> None:
        self._exists = exists_flag

    def exists(self) -> bool:  # pragma: no cover - exercised
        return self._exists

    def __str__(self) -> str:
        return "/fake/path"


def test_validate_yaml_missing_fields_simple() -> None:
    from fastapi import Request
    orig = getattr(mod.validate_yaml, "__wrapped__", mod.validate_yaml)
    req = Request({"type": "http", "method": "POST", "path": "/", "headers": []})
    body = orig({}, req)
    assert body["valid"] is False
    assert "Missing or invalid config type" in body["error"]


def test_validate_yaml_reports_exists(monkeypatch: Any) -> None:
    monkeypatch.setattr(mod, "load_yaml_and_add_lineage", lambda txt: {"data": {"name": "n", "version": "v"}})
    monkeypatch.setattr(mod, "validate_config_payload", lambda t, d: None)
    monkeypatch.setattr(mod, "get_config_path", lambda **k: FakePath(True))

    from fastapi import Request
    orig = getattr(mod.validate_yaml, "__wrapped__", mod.validate_yaml)
    req = Request({"type": "http", "method": "POST", "path": "/", "headers": []})
    body = orig({"type": "interim", "config": "a: b"}, req)
    assert body["valid"] is True
    assert body["exists"] is True


def test_write_yaml_written_and_exists(monkeypatch: Any) -> None:
    monkeypatch.setattr(mod, "load_yaml_and_add_lineage", lambda txt: {"data": {"name": "n", "version": "v"}})
    monkeypatch.setattr(mod, "validate_config_payload", lambda t, d=None: None)

    # exists -> status indicates already exists
    monkeypatch.setattr(mod, "get_config_path", lambda **k: FakePath(True))
    from fastapi import Request
    orig_write = getattr(mod.write_yaml, "__wrapped__", mod.write_yaml)
    req = Request({"type": "http", "method": "POST", "path": "/", "headers": []})
    res_exists = orig_write({"type": "interim", "config": "a: b"}, req)
    assert res_exists["status"] == "exists" or "already" in res_exists.get("message", "")

    # written -> save_config called and returns written path
    fake_path = FakePath(False)
    saved: dict[str, Any] = {}

    def fake_save(cfg: Any, config_path: Any) -> None:
        saved["ok"] = True

    monkeypatch.setattr(mod, "get_config_path", lambda **k: fake_path)
    monkeypatch.setattr(mod, "save_config", fake_save)
    res_written = orig_write({"type": "interim", "config": "a: b"}, req)
    assert res_written["status"] == "written"
    assert "path" in res_written


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


def test_validate_yaml_missing_config_payload() -> None:
    import ml_service.backend.routers.data as data_mod
    from fastapi import Request

    orig = getattr(data_mod.validate_yaml, "__wrapped__", data_mod.validate_yaml)
    req = Request({"type": "http", "method": "POST", "path": "/", "headers": []})
    body = orig({"type": "interim"}, req)
    assert body["valid"] is False
    assert "Missing config payload" in body["error"]


def test_validate_yaml_missing_data_fields(monkeypatch) -> None:
    payload = {"type": "interim", "config": "dummy: yaml"}

    monkeypatch.setattr(
        "ml_service.backend.routers.data.load_yaml_and_add_lineage",
        lambda text: {"data": {}},
    )

    import ml_service.backend.routers.data as data_mod
    from fastapi import Request

    orig = getattr(data_mod.validate_yaml, "__wrapped__", data_mod.validate_yaml)
    req = Request({"type": "http", "method": "POST", "path": "/", "headers": []})
    body = orig(payload, req)
    assert body["valid"] is False
    assert "Missing 'data.name' or 'data.version'" in body["error"]


def test_write_yaml_missing_config_payload_raises() -> None:
    import ml_service.backend.routers.data as data_mod
    from fastapi import Request

    orig = getattr(data_mod.write_yaml, "__wrapped__", data_mod.write_yaml)
    req = Request({"type": "http", "method": "POST", "path": "/", "headers": []})
    with pytest.raises(Exception) as exc:
        orig({"type": "processed"}, req)
    assert "Missing config payload" in str(exc.value)


def test_write_yaml_missing_data_fields_raises(monkeypatch) -> None:
    payload = {"type": "processed", "config": "dummy: yaml"}

    monkeypatch.setattr(
        "ml_service.backend.routers.data.load_yaml_and_add_lineage",
        lambda text: {"data": {}},
    )

    import ml_service.backend.routers.data as data_mod
    from fastapi import Request

    orig = getattr(data_mod.write_yaml, "__wrapped__", data_mod.write_yaml)
    req = Request({"type": "http", "method": "POST", "path": "/", "headers": []})
    with pytest.raises(Exception) as exc:
        orig(payload, req)
    assert "Missing 'data.name' or 'data.version'" in str(exc.value)


def test_write_yaml_missing_type_raises() -> None:
    import ml_service.backend.routers.data as data_mod
    from fastapi import Request

    orig = getattr(data_mod.write_yaml, "__wrapped__", data_mod.write_yaml)
    req = Request({"type": "http", "method": "POST", "path": "/", "headers": []})
    with pytest.raises(Exception) as exc:
        orig({"config": "a: b"}, req)

    assert "Missing or invalid config type" in str(exc.value)
