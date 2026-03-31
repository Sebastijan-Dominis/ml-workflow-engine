import importlib

from fastapi import HTTPException, Request


def test_load_file_missing_path_raises():
    mod = importlib.import_module("ml_service.backend.routers.file_viewer")
    orig = getattr(mod.load_file, "__wrapped__", mod.load_file)
    req = Request({"type": "http", "method": "POST", "path": "/", "headers": []})
    try:
        orig({}, req)
        raised = False
    except HTTPException as e:
        raised = True
        assert e.status_code == 400

    assert raised


def test_load_file_not_found(tmp_path):
    mod = importlib.import_module("ml_service.backend.routers.file_viewer")
    orig = getattr(mod.load_file, "__wrapped__", mod.load_file)
    req = Request({"type": "http", "method": "POST", "path": "/", "headers": []})
    p = tmp_path / "nofile.yaml"
    payload = {"path": str(p)}
    try:
        orig(payload, req)
        raised = False
    except HTTPException as e:
        raised = True
        assert e.status_code == 404

    assert raised


def test_load_yaml_returns_content(tmp_path, monkeypatch):
    mod = importlib.import_module("ml_service.backend.routers.file_viewer")
    orig = getattr(mod.load_file, "__wrapped__", mod.load_file)
    p = tmp_path / "f.yaml"
    p.write_text("a: 1")

    monkeypatch.setattr(mod, "load_yaml", lambda path: {"a": 1})

    req = Request({"type": "http", "method": "POST", "path": "/", "headers": []})
    payload = {"path": str(p)}
    j = orig(payload, req)
    assert j["mode"] == "yaml"
    assert "a:" in j["content"]
    assert j["path"] == str(p)


def test_load_json_returns_content(tmp_path, monkeypatch):
    mod = importlib.import_module("ml_service.backend.routers.file_viewer")
    orig = getattr(mod.load_file, "__wrapped__", mod.load_file)
    p = tmp_path / "f.json"
    p.write_text("{}")

    monkeypatch.setattr(mod, "load_json", lambda path: {"foo": "bar"})

    req = Request({"type": "http", "method": "POST", "path": "/", "headers": []})
    payload = {"path": str(p)}
    j = orig(payload, req)
    assert j["mode"] == "json"
    assert "foo" in j["content"]
    assert j["path"] == str(p)


def test_unsupported_file_type_raises(tmp_path):
    mod = importlib.import_module("ml_service.backend.routers.file_viewer")
    orig = getattr(mod.load_file, "__wrapped__", mod.load_file)
    p = tmp_path / "f.txt"
    p.write_text("hello")
    req = Request({"type": "http", "method": "POST", "path": "/", "headers": []})
    try:
        orig({"path": str(p)}, req)
        raised = False
    except HTTPException as e:
        raised = True
        assert e.status_code == 400

    assert raised
