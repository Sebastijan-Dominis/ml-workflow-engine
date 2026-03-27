import argparse
import types

import pipelines.post_promotion.infer as infer_mod


def test_infer_main_executes_both_stages(monkeypatch):
    args = argparse.Namespace(problem="no_show", segment="segA", snapshot_bindings_id="sb1", logging_level="INFO")
    monkeypatch.setattr(infer_mod, "parse_args", lambda: args)

    called = []

    def fake_execute_inference(*, args, model_metadata, stage, timestamp, path, run_id):
        called.append(stage)

    monkeypatch.setattr(infer_mod, "execute_inference", fake_execute_inference)
    monkeypatch.setattr(infer_mod, "get_model_registry_info", lambda a: types.SimpleNamespace(prod_meta=object(), stage_meta=object()))
    monkeypatch.setattr(infer_mod, "setup_logging", lambda *a, **k: None)

    res = infer_mod.main()

    assert res == 0
    assert "production" in called and "staging" in called


def test_infer_main_prod_only(monkeypatch):
    args = argparse.Namespace(problem="no_show", segment="segA", snapshot_bindings_id="sb1", logging_level="INFO")
    monkeypatch.setattr(infer_mod, "parse_args", lambda: args)

    called = []

    def fake_execute_inference(*, args, model_metadata, stage, timestamp, path, run_id):
        called.append(stage)

    monkeypatch.setattr(infer_mod, "execute_inference", fake_execute_inference)
    monkeypatch.setattr(infer_mod, "get_model_registry_info", lambda a: types.SimpleNamespace(prod_meta=object(), stage_meta=None))
    monkeypatch.setattr(infer_mod, "setup_logging", lambda *a, **k: None)

    res = infer_mod.main()

    assert res == 0
    assert called == ["production"]


def test_infer_main_exception_returns_resolved_code(monkeypatch):
    args = argparse.Namespace(problem="no_show", segment="segA", snapshot_bindings_id="sb1", logging_level="INFO")
    monkeypatch.setattr(infer_mod, "parse_args", lambda: args)
    monkeypatch.setattr(infer_mod, "setup_logging", lambda *a, **k: None)

    def fake_get_info(a):
        raise RuntimeError("boom")

    monkeypatch.setattr(infer_mod, "get_model_registry_info", fake_get_info)
    monkeypatch.setattr(infer_mod, "resolve_exit_code", lambda e: 55)

    res = infer_mod.main()

    assert res == 55
