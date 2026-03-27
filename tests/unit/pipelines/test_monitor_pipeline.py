import argparse
import types

import pipelines.post_promotion.monitor as monitor_mod


def test_monitor_main_raises_when_no_models(monkeypatch):
    args = argparse.Namespace(problem="no_show", segment="segA", inference_run_id="latest", logging_level="INFO")
    monkeypatch.setattr(monitor_mod, "parse_args", lambda: args)
    monkeypatch.setattr(monitor_mod, "setup_logging", lambda *a, **k: None)
    monkeypatch.setattr(monitor_mod, "get_promotion_metrics_info", lambda a: object())
    monkeypatch.setattr(monitor_mod, "get_model_registry_info", lambda a: types.SimpleNamespace(prod_meta=None, stage_meta=None))
    monkeypatch.setattr(monitor_mod, "resolve_exit_code", lambda e: 77)

    res = monitor_mod.main()

    assert res == 77


def test_monitor_main_calls_save_and_compare(monkeypatch):
    args = argparse.Namespace(problem="no_show", segment="segA", inference_run_id="latest", logging_level="INFO")
    monkeypatch.setattr(monitor_mod, "parse_args", lambda: args)
    monkeypatch.setattr(monitor_mod, "setup_logging", lambda *a, **k: None)
    monkeypatch.setattr(monitor_mod, "get_promotion_metrics_info", lambda a: object())
    monkeypatch.setattr(monitor_mod, "get_model_registry_info", lambda a: types.SimpleNamespace(prod_meta=object(), stage_meta=object()))

    def fake_execute_monitoring(*, args, model_metadata, stage, promotion_metrics_info):
        return {"stage": stage, "metrics": {"acc": 0.9 if stage == "production" else 0.85}}

    monkeypatch.setattr(monitor_mod, "execute_monitoring", fake_execute_monitoring)
    monkeypatch.setattr(monitor_mod, "prepare_metadata", lambda **kwargs: {})
    monkeypatch.setattr(monitor_mod, "compare_production_and_staging_performance", lambda p, s: {"delta": 0.05})

    saved = {}

    def fake_save_metadata(metadata, target_dir):
        saved["metadata"] = metadata
        saved["target_dir"] = target_dir

    monkeypatch.setattr(monitor_mod, "save_metadata", fake_save_metadata)

    res = monitor_mod.main()

    assert res == 0
    assert "metadata" in saved
    assert saved["metadata"]["staging_vs_production_comparison"] == {"delta": 0.05}
