"""Integration tests for `pipelines.runners.explain` CLI orchestration."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pipelines.runners.explain as explain_mod


def test_explain_main_success(tmp_path: Path, monkeypatch: Any) -> None:
    experiment_dir = tmp_path / "experiments" / "prob" / "seg" / "v1" / "exp1"
    train_dir = experiment_dir / "training" / "train1"
    train_dir.mkdir(parents=True)

    def fake_get_snapshot_path(sid, parent):
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
        top_k=None,
    )

    monkeypatch.setattr(explain_mod, "parse_args", lambda: args)
    monkeypatch.setattr(explain_mod, "get_snapshot_path", fake_get_snapshot_path)
    monkeypatch.setattr(explain_mod, "bootstrap_logging", lambda *a, **k: None)
    monkeypatch.setattr(explain_mod, "add_file_handler", lambda *a, **k: None)

    model_cfg = SimpleNamespace(
        explainability=SimpleNamespace(enabled=True, top_k=10),
        algorithm=SimpleNamespace(name="catboost"),
    )

    monkeypatch.setattr(explain_mod, "load_and_validate_config", lambda *a, **k: model_cfg)
    monkeypatch.setattr(explain_mod, "add_config_hash", lambda cfg: cfg)
    monkeypatch.setattr(explain_mod, "validate_lineage_integrity", lambda *a, **k: None)
    monkeypatch.setattr(explain_mod, "validate_reproducibility", lambda *a, **k: None)
    monkeypatch.setattr(explain_mod, "validate_pipeline_cfg", lambda *a, **k: "pipeline-hash")
    monkeypatch.setattr(explain_mod, "validate_model_and_pipeline", lambda *a, **k: SimpleNamespace())

    class DummyOutput:
        explainability_metrics: dict[str, Any]
        feature_lineage: list[Any]

        def __init__(self) -> None:
            self.explainability_metrics = {"f": 1.0}
            self.feature_lineage = []

    class DummyExplainer:
        def explain(self, model_cfg, train_dir, top_k):
            return DummyOutput()

    monkeypatch.setattr(explain_mod, "get_explainer", lambda key: DummyExplainer())

    persisted: dict[str, Any] = {}

    def fake_persist(*args, **kwargs):
        persisted["called"] = True

    monkeypatch.setattr(explain_mod, "persist_explainability_run", fake_persist)

    rc = explain_mod.main()
    assert rc == 0
    assert persisted.get("called") is True
