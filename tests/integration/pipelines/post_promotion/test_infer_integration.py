"""Integration tests for `pipelines.post_promotion.infer` CLI."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pipelines.post_promotion.infer as infer_mod


def test_infer_calls_execute_for_prod_and_stage(monkeypatch: Any, tmp_path: Path) -> None:
    args = argparse.Namespace(problem="prob", segment="seg", snapshot_bindings_id="sb", logging_level="INFO")
    monkeypatch.setattr(infer_mod, "parse_args", lambda: args)
    monkeypatch.setattr(infer_mod, "setup_logging", lambda *a, **k: None)

    prod_meta = SimpleNamespace(meta="prod")
    stage_meta = SimpleNamespace(meta="stage")

    monkeypatch.setattr(infer_mod, "get_model_registry_info", lambda a: SimpleNamespace(prod_meta=prod_meta, stage_meta=stage_meta))

    calls: list[dict] = []

    def fake_execute(args, model_metadata, stage, timestamp, path, run_id):
        calls.append({"stage": stage, "path": Path(path)})

    monkeypatch.setattr(infer_mod, "execute_inference", fake_execute)

    rc = infer_mod.main()
    assert rc == 0
    assert any(c["stage"] == "production" for c in calls)
    assert any(c["stage"] == "staging" for c in calls)


def test_infer_no_models_noop(monkeypatch: Any) -> None:
    args = argparse.Namespace(problem="prob", segment="seg", snapshot_bindings_id="sb", logging_level="INFO")
    monkeypatch.setattr(infer_mod, "parse_args", lambda: args)
    monkeypatch.setattr(infer_mod, "setup_logging", lambda *a, **k: None)

    monkeypatch.setattr(infer_mod, "get_model_registry_info", lambda a: SimpleNamespace(prod_meta=None, stage_meta=None))

    called = False

    def fake_execute(*a, **k):
        nonlocal called
        called = True

    monkeypatch.setattr(infer_mod, "execute_inference", fake_execute)

    rc = infer_mod.main()
    assert rc == 0
    assert called is False
