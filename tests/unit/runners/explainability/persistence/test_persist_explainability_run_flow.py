"""Unit tests for explainability persistence orchestration."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest
from ml.runners.explainability.persistence import (
    persist_explainability_run as persist_module,
)

pytestmark = pytest.mark.unit


class _ArtifactsValidationStub:
    """Validation result stub exposing ``model_dump`` for metadata assembly."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.model_dump_kwargs: dict[str, Any] | None = None

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        """Return payload and track dump kwargs for assertions."""
        self.model_dump_kwargs = kwargs
        return self.payload


def _model_cfg_stub() -> Any:
    """Build minimal training config stub consumed by persistence helper."""
    return SimpleNamespace(
        target=SimpleNamespace(name="is_canceled"),
        problem="cancellation",
        segment=SimpleNamespace(name="city_hotel"),
        version="v2",
        meta=SimpleNamespace(config_hash="cfg-hash-42"),
        training=SimpleNamespace(hardware={"task_type": "CPU"}),
    )


def test_persist_explainability_run_minimal_path_without_optional_tables(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persist metadata/runtime with only mandatory model artifacts present."""
    explain_run_dir = tmp_path / "explain" / "run-1"
    artifacts = SimpleNamespace(
        model_hash="model-hash",
        model_path="model.cbm",
        pipeline_path="pipeline.joblib",
        pipeline_hash=None,
    )
    explainability_metrics = SimpleNamespace(
        top_k_feature_importances=None,
        top_k_shap_importances=None,
    )
    captured_raw: dict[str, Any] = {}
    save_metadata_calls: list[dict[str, Any]] = []
    runtime_calls: list[dict[str, Any]] = []
    hash_calls: list[Path] = []
    csv_calls: list[tuple[Path, str]] = []
    validation_stub = _ArtifactsValidationStub(payload={"model_path": "model.cbm", "model_hash": "model-hash"})

    def _validate_artifacts(raw: dict[str, Any]) -> _ArtifactsValidationStub:
        captured_raw.update(raw)
        return validation_stub

    def _save_metadata(metadata: dict[str, Any], target_dir: Path) -> None:
        save_metadata_calls.append({"metadata": metadata, "target_dir": target_dir})

    def _save_runtime_snapshot(**kwargs: Any) -> None:
        runtime_calls.append(kwargs)

    def _hash_artifact(path: Path) -> str:
        hash_calls.append(path)
        return "unexpected-hash"

    def _save_metrics_csv(metrics, *, target_file: Path, name: str) -> None:
        csv_calls.append((target_file, name))

    monkeypatch.setattr(persist_module, "validate_explainability_artifacts", _validate_artifacts)
    monkeypatch.setattr(persist_module, "save_metadata", _save_metadata)
    monkeypatch.setattr(persist_module, "save_runtime_snapshot", _save_runtime_snapshot)
    monkeypatch.setattr(persist_module, "hash_artifact", _hash_artifact)
    monkeypatch.setattr(persist_module, "save_metrics_csv", _save_metrics_csv)

    persist_module.persist_explainability_run(
        _model_cfg_stub(),  # type: ignore[arg-type]
        explain_run_id="explain-1",
        train_run_id="train-1",
        experiment_dir=Path("experiments") / "snapshot-88",
        explain_run_dir=explain_run_dir,
        explainability_metrics=explainability_metrics,  # type: ignore[arg-type]
        feature_lineage=cast(list[Any], [SimpleNamespace(model_dump=lambda: {"name": "lead_time"})]),
        start_time=12.34,
        timestamp="20260306T120000",
        artifacts=artifacts,  # type: ignore[arg-type]
        pipeline_cfg_hash="pipe-hash",
        top_k=15,
    )

    assert captured_raw == {"model_hash": "model-hash", "model_path": "model.cbm"}
    assert validation_stub.model_dump_kwargs == {"exclude_none": True}
    assert csv_calls == []
    assert hash_calls == []
    assert len(save_metadata_calls) == 1
    metadata = save_metadata_calls[0]["metadata"]
    assert metadata["run_identity"]["snapshot_id"] == "snapshot-88"
    assert metadata["artifacts"] == {"model_path": "model.cbm", "model_hash": "model-hash"}
    assert metadata["top_k"] == 15
    assert save_metadata_calls[0]["target_dir"] == explain_run_dir
    assert runtime_calls == [
        {
            "target_dir": explain_run_dir,
            "timestamp": "20260306T120000",
            "hardware_info": {"task_type": "CPU"},
            "start_time": 12.34,
        }
    ]


def test_persist_explainability_run_full_path_with_pipeline_and_top_k_tables(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persist optional pipeline and top-k artifact metadata with hashed CSV files."""
    explain_run_dir = tmp_path / "explain" / "run-2"
    feature_df = pd.DataFrame({"feature": ["adr"], "value": [0.7]})
    shap_df = pd.DataFrame({"feature": ["lead_time"], "value": [0.9]})
    explainability_metrics = SimpleNamespace(
        top_k_feature_importances=feature_df,
        top_k_shap_importances=shap_df,
    )
    artifacts = SimpleNamespace(
        model_hash="model-hash",
        model_path="model.cbm",
        pipeline_path="pipeline.joblib",
        pipeline_hash="pipe-hash-art",
    )
    csv_calls: list[dict[str, Any]] = []
    hash_calls: list[Path] = []
    save_metadata_calls: list[dict[str, Any]] = []
    validation_stub = _ArtifactsValidationStub(payload={"validated": True})

    def _save_metrics_csv_2(metrics, *, target_file: Path, name: str) -> None:
        csv_calls.append({"metrics": metrics, "target_file": target_file, "name": name})

    def _hash_artifact_2(path: Path) -> str:
        hash_calls.append(path)
        return f"hash::{path.name}"

    def _validate_artifacts_2(raw: dict[str, Any]) -> _ArtifactsValidationStub:
        return validation_stub

    def _save_metadata_2(metadata: dict[str, Any], target_dir: Path) -> None:
        save_metadata_calls.append({"metadata": metadata, "target_dir": target_dir})

    monkeypatch.setattr(persist_module, "save_metrics_csv", _save_metrics_csv_2)
    monkeypatch.setattr(persist_module, "hash_artifact", _hash_artifact_2)
    monkeypatch.setattr(persist_module, "validate_explainability_artifacts", _validate_artifacts_2)
    monkeypatch.setattr(persist_module, "save_metadata", _save_metadata_2)
    monkeypatch.setattr(persist_module, "save_runtime_snapshot", lambda **kwargs: None)

    persist_module.persist_explainability_run(
        _model_cfg_stub(),  # type: ignore[arg-type]
        explain_run_id="explain-2",
        train_run_id="train-2",
        experiment_dir=Path("experiments") / "snapshot-99",
        explain_run_dir=explain_run_dir,
        explainability_metrics=explainability_metrics,  # type: ignore[arg-type]
        feature_lineage=cast(list[Any], [SimpleNamespace(model_dump=lambda: {"name": "adr"})]),
        start_time=1.0,
        timestamp="20260306T120001",
        artifacts=artifacts,  # type: ignore[arg-type]
        pipeline_cfg_hash="pipe-hash-runtime",
        top_k=20,
    )

    assert len(csv_calls) == 2
    assert csv_calls[0]["metrics"] is feature_df
    assert csv_calls[0]["target_file"] == explain_run_dir / "top_k_feature_importances.csv"
    assert csv_calls[0]["name"] == "Feature importances"
    assert csv_calls[1]["metrics"] is shap_df
    assert csv_calls[1]["target_file"] == explain_run_dir / "top_k_shap_importances.csv"
    assert csv_calls[1]["name"] == "SHAP importances"

    assert hash_calls == [
        explain_run_dir / "top_k_feature_importances.csv",
        explain_run_dir / "top_k_shap_importances.csv",
    ]
    assert validation_stub.model_dump_kwargs == {"exclude_none": True}

    metadata = save_metadata_calls[0]["metadata"]
    assert metadata["artifacts"] == {"validated": True}
    assert metadata["config_fingerprint"] == {
        "config_hash": "cfg-hash-42",
        "pipeline_cfg_hash": "pipe-hash-runtime",
    }
