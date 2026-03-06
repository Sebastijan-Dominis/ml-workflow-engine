"""Unit tests for explainability persistence orchestration."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

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

    monkeypatch.setattr(
        persist_module,
        "validate_explainability_artifacts",
        lambda raw: captured_raw.update(raw) or validation_stub,
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
    monkeypatch.setattr(
        persist_module,
        "hash_artifact",
        lambda path: hash_calls.append(path) or "unexpected-hash",
    )
    monkeypatch.setattr(
        persist_module,
        "save_metrics_csv",
        lambda metrics, *, target_file, name: csv_calls.append((target_file, name)),
    )

    persist_module.persist_explainability_run(
        _model_cfg_stub(),  # type: ignore[arg-type]
        explain_run_id="explain-1",
        train_run_id="train-1",
        experiment_dir=Path("experiments") / "snapshot-88",
        explain_run_dir=explain_run_dir,
        explainability_metrics=explainability_metrics,  # type: ignore[arg-type]
        feature_lineage=[SimpleNamespace(model_dump=lambda: {"name": "lead_time"})],  # type: ignore[arg-type]
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

    monkeypatch.setattr(
        persist_module,
        "save_metrics_csv",
        lambda metrics, *, target_file, name: csv_calls.append(
            {"metrics": metrics, "target_file": target_file, "name": name}
        ),
    )
    monkeypatch.setattr(
        persist_module,
        "hash_artifact",
        lambda path: hash_calls.append(path) or f"hash::{path.name}",
    )
    monkeypatch.setattr(
        persist_module,
        "validate_explainability_artifacts",
        lambda raw: validation_stub,
    )
    monkeypatch.setattr(
        persist_module,
        "save_metadata",
        lambda metadata, target_dir: save_metadata_calls.append(
            {"metadata": metadata, "target_dir": target_dir}
        ),
    )
    monkeypatch.setattr(persist_module, "save_runtime_snapshot", lambda **kwargs: None)

    persist_module.persist_explainability_run(
        _model_cfg_stub(),  # type: ignore[arg-type]
        explain_run_id="explain-2",
        train_run_id="train-2",
        experiment_dir=Path("experiments") / "snapshot-99",
        explain_run_dir=explain_run_dir,
        explainability_metrics=explainability_metrics,  # type: ignore[arg-type]
        feature_lineage=[SimpleNamespace(model_dump=lambda: {"name": "adr"})],  # type: ignore[arg-type]
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
