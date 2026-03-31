import importlib

from fastapi import Request


def test_generate_cols_for_row_id_fingerprint_calls_execute(monkeypatch):
    mod = importlib.import_module("ml_service.backend.routers.scripts")

    recorded = {}

    def fake_execute(module_path, payload, boolean_args):
        recorded["module_path"] = module_path
        recorded["payload"] = payload
        recorded["boolean_args"] = boolean_args
        return {"ok": True, "module": module_path}

    monkeypatch.setattr(mod, "execute_script", fake_execute)

    orig = getattr(mod.generate_cols_for_row_id_fingerprint, "__wrapped__", mod.generate_cols_for_row_id_fingerprint)
    req = Request({"type": "http", "method": "POST", "path": "/", "headers": []})
    payload = {"x": 1}
    res = orig(payload, req)

    assert recorded["module_path"] == "scripts.generators.generate_cols_for_row_id_fingerprint"
    assert recorded["payload"] == payload
    assert recorded["boolean_args"] == []
    assert res["ok"] is True


def test_generate_fake_data_passes_boolean_args(monkeypatch):
    mod = importlib.import_module("ml_service.backend.routers.scripts")

    seen = {}

    def fake_execute(module_path, payload, boolean_args):
        seen["module_path"] = module_path
        seen["payload"] = payload
        seen["boolean_args"] = boolean_args
        return {"ran": True}

    monkeypatch.setattr(mod, "execute_script", fake_execute)

    orig = getattr(mod.generate_fake_data, "__wrapped__", mod.generate_fake_data)
    req = Request({"type": "http", "method": "POST", "path": "/", "headers": []})
    payload = {"foo": "bar"}
    res = orig(payload, req)

    assert seen["module_path"] == "scripts.generators.generate_fake_data"
    assert "include_old" in seen["boolean_args"]
    assert res["ran"] is True


def test_other_script_endpoints_call_execute(monkeypatch):
    mod = importlib.import_module("ml_service.backend.routers.scripts")

    called = []

    def fake_execute(module_path, payload, boolean_args):
        called.append((module_path, boolean_args))
        return {"ok": True}

    monkeypatch.setattr(mod, "execute_script", fake_execute)

    for fn_name, _expect_module in [
        ("generate_operator_hash", "scripts.generators.generate_operator_hash"),
        ("generate_snapshot_binding", "scripts.generators.generate_snapshot_binding"),
        ("check_import_layers", "scripts.quality.check_import_layers"),
        ("check_naming_conventions", "scripts.quality.check_naming_conventions"),
    ]:
        orig = getattr(getattr(mod, fn_name), "__wrapped__", getattr(mod, fn_name))
        req = Request({"type": "http", "method": "POST", "path": "/", "headers": []})
        res = orig({"ok": True}, req)
        assert res["ok"] is True

    # ensure all expected module paths were called
    called_modules = {m for m, _ in called}
    assert "scripts.generators.generate_operator_hash" in called_modules
    assert "scripts.generators.generate_snapshot_binding" in called_modules
    assert "scripts.quality.check_import_layers" in called_modules
    assert "scripts.quality.check_naming_conventions" in called_modules
