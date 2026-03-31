import importlib

from fastapi import Request


def test_register_raw_snapshot_calls_execute_pipeline(monkeypatch):
    mod = importlib.import_module("ml_service.backend.routers.pipelines")

    recorded = {}

    def fake_execute(module_path, payload, boolean_args):
        recorded["module_path"] = module_path
        recorded["payload"] = payload
        recorded["boolean_args"] = boolean_args
        return {"ok": True, "module": module_path}

    monkeypatch.setattr(mod, "execute_pipeline", fake_execute)

    orig = getattr(mod.register_raw_snapshot, "__wrapped__", mod.register_raw_snapshot)
    req = Request({"type": "http", "method": "POST", "path": "/", "headers": []})
    payload = {"snap": True}
    res = orig(payload, req)

    assert recorded["module_path"] == "pipelines.data.register_raw_snapshot"
    assert recorded["payload"] == payload
    assert recorded["boolean_args"] == []
    assert res["ok"] is True


def test_search_passes_boolean_args(monkeypatch):
    mod = importlib.import_module("ml_service.backend.routers.pipelines")

    seen = {}

    def fake_execute(module_path, payload, boolean_args):
        seen["module_path"] = module_path
        seen["payload"] = payload
        seen["boolean_args"] = boolean_args
        return {"ran": True}

    monkeypatch.setattr(mod, "execute_pipeline", fake_execute)

    orig = getattr(mod.search, "__wrapped__", mod.search)
    req = Request({"type": "http", "method": "POST", "path": "/", "headers": []})
    payload = {"q": 1}
    res = orig(payload, req)

    assert seen["module_path"] == "pipelines.search.search"
    assert "strict" in seen["boolean_args"]
    assert res["ran"] is True
