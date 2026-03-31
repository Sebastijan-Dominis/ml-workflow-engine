"""Integration tests for `pipelines.runners.evaluate` CLI orchestration."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pipelines.runners.evaluate as eval_mod


def test_evaluate_main_success(tmp_path: Path, monkeypatch: Any) -> None:
    experiment_dir = tmp_path / "experiments" / "prob" / "seg" / "v1" / "exp1"
    train_dir = experiment_dir / "training" / "train1"
    train_dir.mkdir(parents=True)

    def fake_get_snapshot_path(sid, parent):
        # parent.name is "training" for the train snapshot
        return train_dir if parent.name == "training" else experiment_dir

    args = argparse.Namespace(
        problem="prob",
        segment="seg",
        version="v1",
        env="default",
        strict=True,
        experiment_id="latest",
        train_id="latest",
        logging_level="INFO",
    )

    monkeypatch.setattr(eval_mod, "parse_args", lambda: args)
    monkeypatch.setattr(eval_mod, "get_snapshot_path", fake_get_snapshot_path)
    monkeypatch.setattr(eval_mod, "bootstrap_logging", lambda *a, **k: None)
    monkeypatch.setattr(eval_mod, "add_file_handler", lambda *a, **k: None)

    # Minimal model config-like object used by the runner
    model_cfg = SimpleNamespace(task=SimpleNamespace(type="classification"), algorithm=SimpleNamespace(name="catboost"))

    monkeypatch.setattr(eval_mod, "load_and_validate_config", lambda *a, **k: model_cfg)
    monkeypatch.setattr(eval_mod, "add_config_hash", lambda cfg: cfg)
    monkeypatch.setattr(eval_mod, "validate_lineage_integrity", lambda *a, **k: None)
    monkeypatch.setattr(eval_mod, "validate_reproducibility", lambda *a, **k: None)
    monkeypatch.setattr(eval_mod, "validate_pipeline_cfg", lambda *a, **k: "pipeline-hash")
    monkeypatch.setattr(eval_mod, "validate_model_and_pipeline", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(eval_mod, "validate_threshold", lambda *a, **k: 0.5)

    class DummyOutput:
        prediction_dfs: dict[str, Any]
        lineage: list[Any]

        def __init__(self) -> None:
            self.metrics = {"acc": {"value": 0.9}}
            self.prediction_dfs = {}
            self.lineage = []

    class DummyEvaluator:
        def evaluate(self, model_cfg, strict, best_threshold, train_dir):
            return DummyOutput()

    monkeypatch.setattr(eval_mod, "get_evaluator", lambda key: DummyEvaluator())

    persisted: dict[str, Any] = {}

    def fake_persist(*args, **kwargs):
        persisted["called"] = True
        persisted["args"] = args
        persisted["kwargs"] = kwargs

    monkeypatch.setattr(eval_mod, "persist_evaluation_run", fake_persist)

    rc = eval_mod.main()

    assert rc == 0
    assert persisted.get("called") is True
