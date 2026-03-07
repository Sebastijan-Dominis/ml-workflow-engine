"""E2E-style smoke tests for the evaluation CLI boundary."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pipelines.runners import evaluate as evaluate_module

pytestmark = pytest.mark.e2e


def test_evaluate_main_executes_end_to_end_control_flow_with_cli_args(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Exercise parse_args + main using realistic argv and successful execution flow."""
    experiment_dir = tmp_path / "experiments" / "no_show" / "global" / "v1" / "exp_200"
    train_dir = experiment_dir / "training" / "train_200"
    train_dir.mkdir(parents=True)

    monkeypatch.setattr(
        evaluate_module.sys,
        "argv",
        [
            "evaluate",
            "--problem",
            "no_show",
            "--segment",
            "global",
            "--version",
            "v1",
            "--env",
            "test",
            "--strict",
            "false",
            "--experiment-id",
            "exp_200",
            "--train-id",
            "train_200",
            "--logging-level",
            "INFO",
        ],
    )

    monkeypatch.setattr(evaluate_module, "iso_no_colon", lambda _dt: "20260307T210000")
    monkeypatch.setattr(evaluate_module, "uuid4", lambda: SimpleNamespace(hex="abcdef0123456789"))
    monkeypatch.setattr(evaluate_module, "bootstrap_logging", lambda level: None)

    file_handlers: list[Path] = []
    monkeypatch.setattr(evaluate_module, "add_file_handler", lambda path, level: file_handlers.append(path))

    def _snapshot(snapshot_id: str, parent_dir: Path) -> Path:
        if parent_dir.name == "v1":
            return experiment_dir
        return train_dir

    monkeypatch.setattr(evaluate_module, "get_snapshot_path", _snapshot)

    model_cfg = SimpleNamespace(task=SimpleNamespace(type="classification"))
    monkeypatch.setattr(evaluate_module, "load_and_validate_config", lambda *args, **kwargs: model_cfg)
    monkeypatch.setattr(evaluate_module, "add_config_hash", lambda cfg: cfg)
    monkeypatch.setattr(evaluate_module, "validate_lineage_integrity", lambda train_dir_arg, cfg: None)
    monkeypatch.setattr(evaluate_module, "validate_reproducibility", lambda runtime_path: None)
    monkeypatch.setattr(evaluate_module, "validate_pipeline_cfg", lambda metadata_path, cfg: "pipeline_hash")
    monkeypatch.setattr(evaluate_module, "validate_model_and_pipeline", lambda train_dir_arg: SimpleNamespace())
    monkeypatch.setattr(evaluate_module, "validate_threshold", lambda task, metrics_path: 0.4)

    output = SimpleNamespace(
        metrics={"test": {"auc": 0.82}},
        prediction_dfs=SimpleNamespace(test_df="dummy"),
        lineage=[SimpleNamespace(name="feature_a")],
    )
    evaluator = SimpleNamespace(evaluate=lambda **kwargs: output)
    monkeypatch.setattr(evaluate_module, "get_evaluator", lambda key: evaluator)

    persisted: dict[str, object] = {}
    monkeypatch.setattr(
        evaluate_module,
        "persist_evaluation_run",
        lambda cfg, **kwargs: persisted.update(kwargs),
    )

    exit_code = evaluate_module.main()

    assert exit_code == 0
    assert file_handlers
    assert file_handlers[0].name == "evaluation.log"
    assert persisted["eval_run_id"] == "20260307T210000_abcdef01"
    assert persisted["train_run_id"] == "train_200"
    assert persisted["metrics"] == {"test": {"auc": 0.82}}


def test_evaluate_main_maps_failures_to_resolved_exit_code(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Return resolver-mapped non-zero code when runtime evaluation dependencies fail."""
    experiment_dir = tmp_path / "experiments" / "no_show" / "global" / "v1" / "exp_201"
    train_dir = experiment_dir / "training" / "train_201"
    train_dir.mkdir(parents=True)

    monkeypatch.setattr(
        evaluate_module.sys,
        "argv",
        [
            "evaluate",
            "--problem",
            "no_show",
            "--segment",
            "global",
            "--version",
            "v1",
            "--experiment-id",
            "exp_201",
            "--train-id",
            "train_201",
        ],
    )

    monkeypatch.setattr(evaluate_module, "iso_no_colon", lambda _dt: "20260307T210500")
    monkeypatch.setattr(evaluate_module, "uuid4", lambda: SimpleNamespace(hex="0011223344556677"))
    monkeypatch.setattr(evaluate_module, "bootstrap_logging", lambda level: None)
    monkeypatch.setattr(evaluate_module, "add_file_handler", lambda path, level: None)
    monkeypatch.setattr(
        evaluate_module,
        "get_snapshot_path",
        lambda snapshot_id, parent_dir: experiment_dir if parent_dir.name == "v1" else train_dir,
    )

    cfg = SimpleNamespace(task=SimpleNamespace(type="classification"))
    monkeypatch.setattr(evaluate_module, "load_and_validate_config", lambda *args, **kwargs: cfg)
    monkeypatch.setattr(evaluate_module, "add_config_hash", lambda model_cfg: model_cfg)

    err = RuntimeError("lineage mismatch")

    def _raise_lineage(*_args: Any, **_kwargs: Any) -> None:
        raise err

    monkeypatch.setattr(evaluate_module, "validate_lineage_integrity", _raise_lineage)
    monkeypatch.setattr(evaluate_module, "resolve_exit_code", lambda e: 41 if e is err else 99)

    assert evaluate_module.main() == 41
