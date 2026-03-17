"""Unit tests for evaluation-run persistence orchestration."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from ml.exceptions import PersistenceError
from ml.runners.evaluation.persistence import persist_evaluation_run as persist_module

pytestmark = pytest.mark.unit


class _ModelDumpStub:
    """Simple stub exposing ``model_dump`` and recording kwargs."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.model_dump_kwargs: dict[str, Any] | None = None

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        """Return payload while capturing dump options."""
        self.model_dump_kwargs = kwargs
        return self.payload


def _model_cfg_stub() -> Any:
    """Build minimal training-config stub for persistence helper."""
    return SimpleNamespace(
        training=SimpleNamespace(hardware={"task_type": "CPU"}),
    )


def _artifacts_stub(*, with_pipeline: bool) -> Any:
    """Build minimal artifacts stub with optional pipeline fields."""
    return SimpleNamespace(
        model_path="model.cbm",
        model_hash="model-hash",
        pipeline_path="pipeline.joblib" if with_pipeline else None,
        pipeline_hash="pipeline-hash" if with_pipeline else None,
    )


def test_persist_evaluation_run_happy_path_calls_all_persistence_steps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persist metrics/predictions/metadata/runtime with expected wiring and payloads."""
    experiment_dir = tmp_path / "experiments" / "snapshot-55"
    eval_run_dir = tmp_path / "eval" / "run-55"
    metrics_file = experiment_dir / "evaluation" / "eval-55" / "metrics.json"

    hash_calls: list[Path] = []
    save_metadata_calls: list[dict[str, Any]] = []
    runtime_calls: list[dict[str, Any]] = []
    validate_raw: dict[str, Any] = {}

    monkeypatch.setattr(persist_module, "save_metrics", lambda *args, **kwargs: str(metrics_file))
    monkeypatch.setattr(
        persist_module,
        "save_predictions",
        lambda prediction_dfs, target_dir: SimpleNamespace(
            train_predictions_path=Path(target_dir / "predictions_train.parquet").as_posix(),
            val_predictions_path=Path(target_dir / "predictions_val.parquet").as_posix(),
            test_predictions_path=Path(target_dir / "predictions_test.parquet").as_posix(),
        ),
    )
    monkeypatch.setattr(
        persist_module,
        "hash_artifact",
        lambda path: hash_calls.append(path) or f"hash::{path.name}",
    )
    monkeypatch.setattr(
        persist_module,
        "PredictionsPathsAndHashes",
        lambda **kwargs: _ModelDumpStub(payload=kwargs),
    )
    monkeypatch.setattr(
        persist_module,
        "validate_evaluation_artifacts",
        lambda raw: validate_raw.update(raw) or _ModelDumpStub(payload={"validated": True}),
    )
    monkeypatch.setattr(
        persist_module,
        "prepare_metadata",
        lambda **kwargs: {"meta": "ok", "artifacts": kwargs["artifacts"].model_dump()},
    )
    monkeypatch.setattr(
        persist_module,
        "save_metadata",
        lambda metadata, target_dir: save_metadata_calls.append(
            {"metadata": metadata, "target_dir": target_dir}
        ),
    )
    monkeypatch.setattr(
        persist_module,
        "save_runtime_snapshot",
        lambda **kwargs: runtime_calls.append(kwargs),
    )

    persist_module.persist_evaluation_run(
        _model_cfg_stub(),  # type: ignore[arg-type]
        eval_run_id="eval-55",
        train_run_id="train-55",
        experiment_dir=experiment_dir,
        eval_run_dir=eval_run_dir,
        metrics={"val": {"auc": 0.8}},
        prediction_dfs=SimpleNamespace(),  # type: ignore[arg-type]
        feature_lineage=[SimpleNamespace(model_dump=lambda: {"name": "adr"})],  # type: ignore[arg-type]
        start_time=9.5,
        timestamp="20260306T130000",
        artifacts=_artifacts_stub(with_pipeline=True),  # type: ignore[arg-type]
        pipeline_cfg_hash="runtime-pipeline-hash",
    )

    assert hash_calls == [
        metrics_file,
        eval_run_dir / "predictions_train.parquet",
        eval_run_dir / "predictions_val.parquet",
        eval_run_dir / "predictions_test.parquet",
    ]
    assert validate_raw["metrics_path"] == Path(metrics_file).as_posix()
    assert validate_raw["metrics_hash"] == "hash::metrics.json"
    assert validate_raw["pipeline_path"] == "pipeline.joblib"
    assert validate_raw["pipeline_hash"] == "pipeline-hash"
    assert validate_raw["train_predictions_hash"] == "hash::predictions_train.parquet"
    assert validate_raw["val_predictions_hash"] == "hash::predictions_val.parquet"
    assert validate_raw["test_predictions_hash"] == "hash::predictions_test.parquet"

    assert save_metadata_calls == [
        {"metadata": {"meta": "ok", "artifacts": {"validated": True}}, "target_dir": eval_run_dir}
    ]
    assert runtime_calls == [
        {
            "target_dir": eval_run_dir,
            "timestamp": "20260306T130000",
            "hardware_info": {"task_type": "CPU"},
            "start_time": 9.5,
        }
    ]


def test_persist_evaluation_run_wraps_predictions_model_construction_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wrap prediction-path model construction failures as PersistenceError."""
    experiment_dir = tmp_path / "experiments" / "snapshot-77"
    eval_run_dir = tmp_path / "eval" / "run-77"

    monkeypatch.setattr(
        persist_module,
        "save_metrics",
        lambda *args, **kwargs: str(experiment_dir / "evaluation" / "eval-77" / "metrics.json"),
    )
    monkeypatch.setattr(
        persist_module,
        "hash_artifact",
        lambda path: "hash",
    )
    monkeypatch.setattr(
        persist_module,
        "save_predictions",
        lambda prediction_dfs, target_dir: SimpleNamespace(
            train_predictions_path=str(target_dir / "predictions_train.parquet"),
            val_predictions_path=str(target_dir / "predictions_val.parquet"),
            test_predictions_path=str(target_dir / "predictions_test.parquet"),
        ),
    )
    monkeypatch.setattr(
        persist_module,
        "PredictionsPathsAndHashes",
        lambda **kwargs: (_ for _ in ()).throw(ValueError("bad model")),
    )

    save_metadata_calls: list[dict[str, Any]] = []
    runtime_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        persist_module,
        "save_metadata",
        lambda metadata, target_dir: save_metadata_calls.append(
            {"metadata": metadata, "target_dir": target_dir}
        ),
    )
    monkeypatch.setattr(
        persist_module,
        "save_runtime_snapshot",
        lambda **kwargs: runtime_calls.append(kwargs),
    )

    with pytest.raises(
        PersistenceError,
        match="Failed to construct predictions paths and hashes model",
    ) as exc_info:
        persist_module.persist_evaluation_run(
            _model_cfg_stub(),  # type: ignore[arg-type]
            eval_run_id="eval-77",
            train_run_id="train-77",
            experiment_dir=experiment_dir,
            eval_run_dir=eval_run_dir,
            metrics={"val": {"auc": 0.8}},
            prediction_dfs=SimpleNamespace(),  # type: ignore[arg-type]
            feature_lineage=[],  # type: ignore[arg-type]
            start_time=1.0,
            timestamp="20260306T130001",
            artifacts=_artifacts_stub(with_pipeline=False),  # type: ignore[arg-type]
            pipeline_cfg_hash="runtime-pipeline-hash",
        )

    assert isinstance(exc_info.value.__cause__, ValueError)
    assert save_metadata_calls == []
    assert runtime_calls == []
