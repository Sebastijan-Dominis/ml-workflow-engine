"""Unit tests for evaluation-run metadata preparation helper."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from ml.runners.evaluation.persistence import prepare_metadata as prepare_metadata_module

pytestmark = pytest.mark.unit


class _ModelDumpStub:
    """Stub object exposing ``model_dump`` and capturing call kwargs."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.model_dump_kwargs: dict[str, Any] | None = None

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        """Return payload while recording dump options for assertions."""
        self.model_dump_kwargs = kwargs
        return self.payload


def test_prepare_metadata_builds_expected_payload_and_validates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Assemble complete metadata payload and pass it through validation layer."""
    model_cfg = SimpleNamespace(
        target=SimpleNamespace(name="is_canceled"),
        problem="cancellation",
        segment=SimpleNamespace(name="city_hotel"),
        version="v3",
        meta=SimpleNamespace(config_hash="cfg-hash-123"),
    )
    feature_lineage = [
        SimpleNamespace(model_dump=lambda: {"feature": "lead_time"}),
        SimpleNamespace(model_dump=lambda: {"feature": "adr"}),
    ]
    artifacts_stub = _ModelDumpStub(payload={"metrics_path": "metrics.json", "optional": None})
    metadata_stub = _ModelDumpStub(payload={"normalized": True})
    captured_raw: dict[str, Any] = {}

    def _validate(raw: dict[str, Any]) -> _ModelDumpStub:
        captured_raw.update(raw)
        return metadata_stub

    monkeypatch.setattr(prepare_metadata_module, "validate_evaluation_metadata", _validate)

    result = prepare_metadata_module.prepare_metadata(
        model_cfg=model_cfg,  # type: ignore[arg-type]
        eval_run_id="eval-1",
        train_run_id="train-1",
        experiment_dir=Path("experiments") / "cancellation" / "snapshot-42",
        feature_lineage=feature_lineage,  # type: ignore[arg-type]
        artifacts=artifacts_stub,  # type: ignore[arg-type]
        pipeline_cfg_hash="pipe-hash-1",
    )

    assert result == {"normalized": True}
    assert artifacts_stub.model_dump_kwargs == {"exclude_none": True}
    assert metadata_stub.model_dump_kwargs == {"exclude_none": True}
    assert captured_raw == {
        "run_identity": {
            "stage": "evaluation",
            "eval_run_id": "eval-1",
            "train_run_id": "train-1",
            "snapshot_id": "snapshot-42",
            "status": "success",
        },
        "lineage": {
            "feature_lineage": [{"feature": "lead_time"}, {"feature": "adr"}],
            "target_column": "is_canceled",
            "problem": "cancellation",
            "segment": "city_hotel",
            "model_version": "v3",
        },
        "config_fingerprint": {
            "config_hash": "cfg-hash-123",
            "pipeline_cfg_hash": "pipe-hash-1",
        },
        "artifacts": {"metrics_path": "metrics.json", "optional": None},
    }


def test_prepare_metadata_uses_experiment_dir_name_for_snapshot_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Derive snapshot identifier strictly from terminal experiment directory segment."""
    model_cfg = SimpleNamespace(
        target=SimpleNamespace(name="target"),
        problem="problem",
        segment=SimpleNamespace(name="segment"),
        version="v1",
        meta=SimpleNamespace(config_hash="cfg"),
    )
    metadata_stub = _ModelDumpStub(payload={"ok": True})
    raw_payloads: list[dict[str, Any]] = []

    monkeypatch.setattr(
        prepare_metadata_module,
        "validate_evaluation_metadata",
        lambda raw: raw_payloads.append(raw) or metadata_stub,
    )

    prepare_metadata_module.prepare_metadata(
        model_cfg=model_cfg,  # type: ignore[arg-type]
        eval_run_id="eval",
        train_run_id="train",
        experiment_dir=Path("a") / "b" / "c" / "final-snapshot",
        feature_lineage=[],  # type: ignore[arg-type]
        artifacts=_ModelDumpStub(payload={}),  # type: ignore[arg-type]
        pipeline_cfg_hash="pipe",
    )

    assert raw_payloads[0]["run_identity"]["snapshot_id"] == "final-snapshot"
