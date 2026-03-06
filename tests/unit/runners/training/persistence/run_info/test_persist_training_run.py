"""Unit tests for training run persistence orchestration."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from ml.config.schemas.model_cfg import TrainModelConfig
from ml.modeling.models.feature_lineage import FeatureLineage
from ml.runners.training.persistence.run_info import persist_training_run as module

pytestmark = pytest.mark.unit


class _LineageStub:
    """Feature-lineage stub exposing ``model_dump`` for metadata assembly."""

    def __init__(self, payload: dict[str, Any]) -> None:
        """Store payload returned by ``model_dump`` during metadata creation."""
        self._payload = payload

    def model_dump(self) -> dict[str, Any]:
        """Return payload representing one lineage record."""
        return self._payload


def _model_cfg_stub() -> TrainModelConfig:
    """Build minimal ``TrainModelConfig``-compatible object for persistence tests."""
    return cast(
        TrainModelConfig,
        SimpleNamespace(
            target=SimpleNamespace(name="adr"),
            problem="hotel_bookings",
            segment=SimpleNamespace(name="global"),
            version="v1",
            meta=SimpleNamespace(config_hash="cfg-hash"),
            training=SimpleNamespace(hardware=SimpleNamespace(device="cpu")),
        ),
    )


def test_persist_training_run_persists_validated_metadata_metrics_and_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Persist validated metadata, metrics artifact, and runtime snapshot in order."""
    captured: dict[str, Any] = {}

    def _validate_training_metadata(raw: dict[str, Any]) -> Any:
        captured["raw_metadata"] = raw
        return SimpleNamespace(model_dump=lambda **_kwargs: {"validated": True})

    monkeypatch.setattr(module, "validate_training_metadata", _validate_training_metadata)

    def _save_metadata(payload: dict[str, Any], *, target_dir: Path) -> None:
        captured["saved_metadata_payload"] = payload
        captured["saved_metadata_dir"] = target_dir

    monkeypatch.setattr(module, "save_metadata", _save_metadata)

    def _save_metrics(
        metrics: dict[str, float],
        *,
        model_cfg: TrainModelConfig,
        target_run_id: str,
        experiment_dir: Path,
        stage: str,
    ) -> None:
        captured["saved_metrics"] = metrics
        captured["saved_metrics_model_cfg"] = model_cfg
        captured["saved_metrics_run_id"] = target_run_id
        captured["saved_metrics_experiment_dir"] = experiment_dir
        captured["saved_metrics_stage"] = stage

    monkeypatch.setattr(module, "save_metrics", _save_metrics)

    def _save_runtime_snapshot(*, target_dir: Path, timestamp: str, hardware_info: Any, start_time: float) -> None:
        captured["runtime_target_dir"] = target_dir
        captured["runtime_timestamp"] = timestamp
        captured["runtime_hardware"] = hardware_info
        captured["runtime_start_time"] = start_time

    monkeypatch.setattr(module, "save_runtime_snapshot", _save_runtime_snapshot)

    cfg = _model_cfg_stub()
    experiment_dir = tmp_path / "experiments" / "exp_001"
    train_run_dir = tmp_path / "runs" / "train_001"

    module.persist_training_run(
        cfg,
        train_run_id="train_001",
        experiment_dir=experiment_dir,
        train_run_dir=train_run_dir,
        start_time=123.45,
        timestamp="2026-03-07T12:00:00",
        feature_lineage=cast(list[FeatureLineage], [_LineageStub({"feature_set": "booking_context_features"})]),
        metrics={"rmse": 0.99},
        model_hash="model-hash",
        pipeline_hash="pipeline-hash",
        model_path=train_run_dir / "model.joblib",
        pipeline_path=train_run_dir / "pipeline.joblib",
        pipeline_cfg_hash="pipeline-cfg-hash",
    )

    raw = captured["raw_metadata"]
    assert raw["run_identity"]["stage"] == "training"
    assert raw["run_identity"]["snapshot_id"] == "exp_001"
    assert raw["lineage"]["target_column"] == "adr"
    assert raw["config_fingerprint"]["pipeline_cfg_hash"] == "pipeline-cfg-hash"
    assert raw["artifacts"]["pipeline_hash"] == "pipeline-hash"
    assert raw["artifacts"]["pipeline_path"].endswith("pipeline.joblib")

    assert captured["saved_metadata_payload"] == {"validated": True}
    assert captured["saved_metadata_dir"] == train_run_dir

    assert captured["saved_metrics"] == {"rmse": 0.99}
    assert captured["saved_metrics_model_cfg"] is cfg
    assert captured["saved_metrics_run_id"] == "train_001"
    assert captured["saved_metrics_experiment_dir"] == experiment_dir
    assert captured["saved_metrics_stage"] == "training"

    assert captured["runtime_target_dir"] == train_run_dir
    assert captured["runtime_timestamp"] == "2026-03-07T12:00:00"
    assert captured["runtime_hardware"] is cfg.training.hardware
    assert captured["runtime_start_time"] == 123.45


def test_persist_training_run_omits_pipeline_fields_when_not_fully_available(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Do not include pipeline metadata fields unless all pipeline values are present."""
    captured: dict[str, Any] = {}

    def _validate_training_metadata(raw: dict[str, Any]) -> Any:
        captured["raw_metadata"] = raw
        return SimpleNamespace(model_dump=lambda **_kwargs: {})

    monkeypatch.setattr(
        module,
        "validate_training_metadata",
        _validate_training_metadata,
    )
    monkeypatch.setattr(module, "save_metadata", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "save_metrics", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "save_runtime_snapshot", lambda *_args, **_kwargs: None)

    cfg = _model_cfg_stub()

    module.persist_training_run(
        cfg,
        train_run_id="train_002",
        experiment_dir=tmp_path / "experiments" / "exp_002",
        train_run_dir=tmp_path / "runs" / "train_002",
        start_time=1.0,
        timestamp="2026-03-07T13:00:00",
        feature_lineage=cast(list[FeatureLineage], [_LineageStub({"feature_set": "booking_context_features"})]),
        metrics={"rmse": 1.1},
        model_hash="model-hash",
        pipeline_hash=None,
        model_path=tmp_path / "runs" / "train_002" / "model.joblib",
        pipeline_path=tmp_path / "runs" / "train_002" / "pipeline.joblib",
        pipeline_cfg_hash="pipeline-cfg-hash",
    )

    raw = captured["raw_metadata"]
    assert "pipeline_cfg_hash" not in raw["config_fingerprint"]
    assert "pipeline_hash" not in raw["artifacts"]
    assert "pipeline_path" not in raw["artifacts"]
