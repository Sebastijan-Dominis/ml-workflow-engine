"""Integration tests for `pipelines.post_promotion.monitor` CLI."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pipelines.post_promotion.monitor as monitor_mod


def test_monitor_main_success(monkeypatch: Any, tmp_path: Path) -> None:
    args = argparse.Namespace(problem="prob", segment="seg", inference_run_id="latest", logging_level="INFO")
    monkeypatch.setattr(monitor_mod, "parse_args", lambda: args)
    monkeypatch.setattr(monitor_mod, "setup_logging", lambda *a, **k: None)

    monkeypatch.setattr(monitor_mod, "get_promotion_metrics_info", lambda a: {"metrics": True})

    prod_meta = SimpleNamespace(meta="prod")
    stage_meta = SimpleNamespace(meta="stage")
    monkeypatch.setattr(monitor_mod, "get_model_registry_info", lambda a: SimpleNamespace(prod_meta=prod_meta, stage_meta=stage_meta))

    monkeypatch.setattr(monitor_mod, "execute_monitoring", lambda *a, **k: {"monitor": True})
    monkeypatch.setattr(monitor_mod, "compare_production_and_staging_performance", lambda p, s: {"delta": 0.1})
    monkeypatch.setattr(monitor_mod, "prepare_metadata", lambda **k: {"meta": True})

    saved: dict[str, Any] = {}

    def fake_save_metadata(obj, target_dir):
        saved["obj"] = obj
        saved["target_dir"] = Path(target_dir)

    monkeypatch.setattr(monitor_mod, "save_metadata", fake_save_metadata)

    rc = monitor_mod.main()
    assert rc == 0
    assert "staging_vs_production_comparison" in saved["obj"]


def test_monitor_main_no_models_returns_error_code(monkeypatch: Any) -> None:
    args = argparse.Namespace(problem="prob", segment="seg", inference_run_id="latest", logging_level="INFO")
    monkeypatch.setattr(monitor_mod, "parse_args", lambda: args)
    monkeypatch.setattr(monitor_mod, "setup_logging", lambda *a, **k: None)

    monkeypatch.setattr(monitor_mod, "get_promotion_metrics_info", lambda a: {"metrics": True})
    monkeypatch.setattr(monitor_mod, "get_model_registry_info", lambda a: SimpleNamespace(prod_meta=None, stage_meta=None))

    monkeypatch.setattr(monitor_mod, "resolve_exit_code", lambda e: 99)

    rc = monitor_mod.main()
    assert rc == 99
