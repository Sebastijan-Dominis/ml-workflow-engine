"""Unit tests for evaluation runner CLI orchestration."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pipelines.runners import evaluate as module

pytestmark = pytest.mark.unit


def test_parse_args_uses_expected_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Parse required args and preserve default optional evaluation flags."""
    monkeypatch.setattr(
        module.sys,
        "argv",
        ["evaluate", "--problem", "no_show", "--segment", "global", "--version", "v1"],
    )

    args = module.parse_args()

    assert args.problem == "no_show"
    assert args.segment == "global"
    assert args.version == "v1"
    assert args.env == "default"
    assert args.strict is True
    assert args.experiment_id == "latest"
    assert args.train_id == "latest"
    assert args.logging_level == "INFO"


def test_main_returns_resolved_code_when_snapshot_lookup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Map experiment/training snapshot resolution failures via exit-code resolver."""
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: Namespace(
            problem="no_show",
            segment="global",
            version="v1",
            env="default",
            strict=True,
            experiment_id="latest",
            train_id="latest",
            logging_level="INFO",
        ),
    )
    monkeypatch.setattr(module, "bootstrap_logging", lambda level: None)

    err = RuntimeError("missing snapshots")
    monkeypatch.setattr(module, "get_snapshot_path", lambda snapshot_id, parent_dir: (_ for _ in ()).throw(err))
    monkeypatch.setattr(module, "resolve_exit_code", lambda e: 27 if e is err else 99)

    code = module.main()

    assert code == 27


def test_main_runs_evaluation_and_persists_outputs_on_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Execute happy-path evaluation flow and persist metrics/predictions metadata."""
    experiment_dir = tmp_path / "experiments" / "no_show" / "global" / "v1" / "exp_1"
    train_dir = experiment_dir / "training" / "train_1"
    train_dir.mkdir(parents=True)

    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: Namespace(
            problem="no_show",
            segment="global",
            version="v1",
            env="test",
            strict=False,
            experiment_id="exp_1",
            train_id="train_1",
            logging_level="DEBUG",
        ),
    )
    monkeypatch.setattr(module, "iso_no_colon", lambda _dt: "20260306T181000")
    monkeypatch.setattr(module, "uuid4", lambda: SimpleNamespace(hex="abcdef0123456789"))
    monkeypatch.setattr(module, "bootstrap_logging", lambda level: None)

    snapshot_calls: list[tuple[str, Path]] = []

    def _fake_get_snapshot_path(snapshot_id: str, parent_dir: Path) -> Path:
        snapshot_calls.append((snapshot_id, parent_dir))
        if parent_dir.name == "v1":
            return experiment_dir
        return train_dir

    monkeypatch.setattr(module, "get_snapshot_path", _fake_get_snapshot_path)

    file_handler_calls: list[Path] = []
    monkeypatch.setattr(module, "add_file_handler", lambda path, level: file_handler_calls.append(path))

    cfg = SimpleNamespace(task=SimpleNamespace(type="binary"))
    monkeypatch.setattr(module, "load_and_validate_config", lambda *args, **kwargs: cfg)
    monkeypatch.setattr(module, "add_config_hash", lambda model_cfg: model_cfg)

    monkeypatch.setattr(module, "validate_lineage_integrity", lambda train_dir_arg, model_cfg: None)
    monkeypatch.setattr(module, "validate_reproducibility", lambda runtime_path: None)
    monkeypatch.setattr(module, "validate_pipeline_cfg", lambda metadata_path, model_cfg: "pipeline-cfg-hash")
    monkeypatch.setattr(module, "validate_model_and_pipeline", lambda train_dir_arg: SimpleNamespace(model_path="m"))
    monkeypatch.setattr(module, "validate_threshold", lambda task, metrics_path: 0.42)

    eval_output = SimpleNamespace(
        metrics={"test": {"auc": 0.81}},
        prediction_dfs=SimpleNamespace(test_df="dummy"),
        lineage=[SimpleNamespace(feature="x")],
    )
    evaluator = SimpleNamespace(evaluate=lambda **kwargs: eval_output)
    monkeypatch.setattr(module, "get_evaluator", lambda key: evaluator)

    persisted: dict[str, Any] = {}
    monkeypatch.setattr(
        module,
        "persist_evaluation_run",
        lambda model_cfg, **kwargs: persisted.update(kwargs),
    )

    code = module.main()

    assert code == 0
    assert snapshot_calls[0][0] == "exp_1"
    assert snapshot_calls[1][0] == "train_1"
    assert file_handler_calls[0].name == "evaluation.log"
    assert persisted["eval_run_id"] == "20260306T181000_abcdef01"
    assert persisted["train_run_id"] == "train_1"
    assert persisted["metrics"] == {"test": {"auc": 0.81}}
    assert persisted["pipeline_cfg_hash"] == "pipeline-cfg-hash"


def test_main_maps_resolve_exit_code_when_evaluator_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Map evaluator runtime failures through shared CLI exit-code resolution."""
    experiment_dir = tmp_path / "experiments" / "no_show" / "global" / "v1" / "exp_2"
    train_dir = experiment_dir / "training" / "train_2"
    train_dir.mkdir(parents=True)

    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: Namespace(
            problem="no_show",
            segment="global",
            version="v1",
            env="default",
            strict=True,
            experiment_id="exp_2",
            train_id="train_2",
            logging_level="INFO",
        ),
    )
    monkeypatch.setattr(module, "iso_no_colon", lambda _dt: "20260306T181500")
    monkeypatch.setattr(module, "uuid4", lambda: SimpleNamespace(hex="0011223344556677"))
    monkeypatch.setattr(module, "bootstrap_logging", lambda level: None)
    monkeypatch.setattr(
        module,
        "get_snapshot_path",
        lambda snapshot_id, parent_dir: experiment_dir if parent_dir.name == "v1" else train_dir,
    )
    monkeypatch.setattr(module, "add_file_handler", lambda path, level: None)

    cfg = SimpleNamespace(task=SimpleNamespace(type="binary"))
    monkeypatch.setattr(module, "load_and_validate_config", lambda *args, **kwargs: cfg)
    monkeypatch.setattr(module, "add_config_hash", lambda model_cfg: model_cfg)
    monkeypatch.setattr(module, "validate_lineage_integrity", lambda train_dir_arg, model_cfg: None)
    monkeypatch.setattr(module, "validate_reproducibility", lambda runtime_path: None)
    monkeypatch.setattr(module, "validate_pipeline_cfg", lambda metadata_path, model_cfg: "h")
    monkeypatch.setattr(module, "validate_model_and_pipeline", lambda train_dir_arg: SimpleNamespace(model_path="m"))
    monkeypatch.setattr(module, "validate_threshold", lambda task, metrics_path: None)

    err = RuntimeError("evaluator failed")

    def _raise(**kwargs: Any) -> Any:
        raise err

    monkeypatch.setattr(module, "get_evaluator", lambda key: SimpleNamespace(evaluate=_raise))
    monkeypatch.setattr(module, "resolve_exit_code", lambda e: 73 if e is err else 99)

    code = module.main()

    assert code == 73
