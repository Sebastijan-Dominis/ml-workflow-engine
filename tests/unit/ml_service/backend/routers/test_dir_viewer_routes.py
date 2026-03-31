import importlib
from typing import Any, cast

from fastapi import HTTPException, Request


def test_load_dir_missing_path_raises():
    mod = importlib.import_module("ml_service.backend.routers.dir_viewer")
    orig = getattr(mod.load_dir, "__wrapped__", mod.load_dir)
    req = Request({"type": "http", "method": "POST", "path": "/", "headers": []})
    try:
        orig({}, req)
        raised = False
    except HTTPException as e:
        raised = True
        assert e.status_code == 400

    assert raised


def test_load_dir_outside_repo_raises(tmp_path):
    mod = importlib.import_module("ml_service.backend.routers.dir_viewer")
    orig = getattr(mod.load_dir, "__wrapped__", mod.load_dir)
    # make repo_root the tmp path so '../' escapes it
    cast(Any, mod).repo_root = str(tmp_path)
    req = Request({"type": "http", "method": "POST", "path": "/", "headers": []})
    try:
        orig({"path": ".."}, req)
        raised = False
    except HTTPException as e:
        raised = True
        assert e.status_code == 403

    assert raised


def test_load_dir_not_found(tmp_path):
    mod = importlib.import_module("ml_service.backend.routers.dir_viewer")
    orig = getattr(mod.load_dir, "__wrapped__", mod.load_dir)
    cast(Any, mod).repo_root = str(tmp_path)
    req = Request({"type": "http", "method": "POST", "path": "/", "headers": []})
    try:
        orig({"path": "nope"}, req)
        raised = False
    except HTTPException as e:
        raised = True
        assert e.status_code == 404

    assert raised


def test_load_dir_success(tmp_path, monkeypatch):
    mod = importlib.import_module("ml_service.backend.routers.dir_viewer")
    orig = getattr(mod.load_dir, "__wrapped__", mod.load_dir)
    # use monkeypatch to set a module attribute safely
    monkeypatch.setattr(cast(Any, mod), "repo_root", str(tmp_path), raising=False)
    d = tmp_path / "sub"
    d.mkdir()
    monkeypatch.setattr(mod, "build_tree", lambda p: {"ok": True})
    req = Request({"type": "http", "method": "POST", "path": "/", "headers": []})
    res = orig({"path": "sub"}, req)
    assert res["tree"] == {"ok": True}
    assert res["path"] == str(d.resolve())
    assert "tree_yaml" in res
