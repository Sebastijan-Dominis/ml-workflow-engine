"""Integration tests for the training runner CLI entrypoint.

These tests exercise `pipelines.runners.train.main` in a controlled way by
monkeypatching filesystem and heavy dependencies so the function's control
flow and persistence handoff can be validated end-to-end without running a
real training job.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pipelines.runners.train as train_mod


def _make_args() -> SimpleNamespace:
    return SimpleNamespace(
        problem="p",
        segment="s",
        version="v",
        snapshot_binding_key=None,
        train_run_id=None,
        env="default",
        strict=True,
        experiment_id=None,
        logging_level="INFO",
        clean_up_failure_management=False,
        overwrite_existing=False,
    )


def test_train_main_success(tmp_path: Path, monkeypatch: Any) -> None:
    """`main()` completes successfully when trainer and persistence behave."""

    args = _make_args()

    # Provide a deterministic experiment directory under tmp_path
    experiment_dir = tmp_path / "experiments" / args.problem / args.segment / args.version / "exp1"
    experiment_dir.mkdir(parents=True)
    (experiment_dir / "search").mkdir()

    monkeypatch.setattr(train_mod, "parse_args", lambda: args)
    monkeypatch.setattr(train_mod, "get_snapshot_path", lambda eid, parent: experiment_dir)

    # Keep config hashing/loader simple
    monkeypatch.setattr(train_mod, "add_config_hash", lambda cfg: cfg)
    fake_cfg = SimpleNamespace(algorithm=SimpleNamespace(value="catboost"))
    monkeypatch.setattr(train_mod, "load_and_validate_config", lambda *a, **k: fake_cfg)

    # Validators are no-ops for the integration test
    monkeypatch.setattr(train_mod, "validate_lineage_integrity", lambda *_: None)
    monkeypatch.setattr(train_mod, "validate_reproducibility", lambda *_: None)
    monkeypatch.setattr(train_mod, "validate_logical_config", lambda *_: None)
    monkeypatch.setattr(train_mod, "validate_pipeline_cfg", lambda *_: None)

    # Fake trainer that returns the minimal expected TrainOutput-like object
    fake_output = SimpleNamespace(model=SimpleNamespace(), pipeline=None, lineage=[], metrics={}, pipeline_cfg_hash=None)

    class FakeTrainer:
        def train(self, *a, **k):
            return fake_output

    monkeypatch.setattr(train_mod, "get_trainer", lambda alg: FakeTrainer())

    # Persist helpers: write a tiny model file and return its path
    def fake_save_model(model, train_run_dir: Path) -> Path:
        train_run_dir.mkdir(parents=True, exist_ok=True)
        p = train_run_dir / "model.bin"
        p.write_text("ok")
        return p

    monkeypatch.setattr(train_mod, "save_model", fake_save_model)
    monkeypatch.setattr(train_mod, "save_pipeline", lambda *a, **k: experiment_dir / "training" / "pipeline.joblib")
    monkeypatch.setattr(train_mod, "persist_training_run", lambda *a, **k: None)
    monkeypatch.setattr(train_mod, "hash_artifact", lambda p: "deadbeef")
    monkeypatch.setattr(train_mod, "delete_failure_management_folder", lambda *a, **k: None)
    monkeypatch.setattr(train_mod, "add_file_handler", lambda *a, **k: None)
    monkeypatch.setattr(train_mod, "bootstrap_logging", lambda *a, **k: None)

    rc = train_mod.main()
    assert rc == 0


def test_train_main_returns_resolve_code_on_exception(tmp_path: Path, monkeypatch: Any) -> None:
    """If the trainer raises, `main()` returns whatever `resolve_exit_code` yields."""

    args = _make_args()
    experiment_dir = tmp_path / "exps" / "p" / "s" / "v" / "exp2"
    experiment_dir.mkdir(parents=True)
    (experiment_dir / "search").mkdir()

    monkeypatch.setattr(train_mod, "parse_args", lambda: args)
    monkeypatch.setattr(train_mod, "get_snapshot_path", lambda eid, parent: experiment_dir)

    monkeypatch.setattr(train_mod, "add_config_hash", lambda cfg: cfg)
    fake_cfg = SimpleNamespace(algorithm=SimpleNamespace(value="catboost"))
    monkeypatch.setattr(train_mod, "load_and_validate_config", lambda *a, **k: fake_cfg)
    monkeypatch.setattr(train_mod, "validate_lineage_integrity", lambda *_: None)
    monkeypatch.setattr(train_mod, "validate_reproducibility", lambda *_: None)
    monkeypatch.setattr(train_mod, "validate_logical_config", lambda *_: None)
    monkeypatch.setattr(train_mod, "validate_pipeline_cfg", lambda *_: None)

    class BrokenTrainer:
        def train(self, *a, **k):
            raise RuntimeError("boom")

    monkeypatch.setattr(train_mod, "get_trainer", lambda alg: BrokenTrainer())
    monkeypatch.setattr(train_mod, "resolve_exit_code", lambda e: 42)
    monkeypatch.setattr(train_mod, "add_file_handler", lambda *a, **k: None)
    monkeypatch.setattr(train_mod, "bootstrap_logging", lambda *a, **k: None)

    rc = train_mod.main()
    assert rc == 42

