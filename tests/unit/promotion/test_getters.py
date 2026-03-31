"""Unit tests for promotion input getter utility helpers."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from ml.exceptions import PersistenceError, UserError
from ml.metadata.schemas.runners.training import TrainingMetadata
from ml.promotion.getters.get import (
    extract_thresholds,
    get_pipeline_cfg_hash,
    get_runners_metadata,
    get_training_conda_env_hash,
)

pytestmark = pytest.mark.unit


def test_extract_thresholds_returns_problem_segment_threshold_mapping() -> None:
    """Select the exact threshold dictionary for the requested problem and segment."""
    thresholds: dict[str, Any] = {
        "cancellation": {
            "city_hotel": {"promotion_metrics": {"sets": ["val"]}},
        }
    }

    result = extract_thresholds(thresholds, "cancellation", "city_hotel")

    assert result == {"promotion_metrics": {"sets": ["val"]}}


def test_extract_thresholds_raises_when_problem_segment_not_found() -> None:
    """Raise UserError when no threshold config exists for requested problem/segment."""
    thresholds: dict[str, Any] = {"cancellation": {}}

    with pytest.raises(UserError, match="No promotion thresholds found"):
        extract_thresholds(thresholds, "cancellation", "city_hotel")


def test_get_pipeline_cfg_hash_returns_hash_from_training_metadata() -> None:
    """Read and return pipeline config hash from validated training metadata object."""
    training_metadata = cast(
        TrainingMetadata,
        SimpleNamespace(config_fingerprint=SimpleNamespace(pipeline_cfg_hash="cfg-hash-123")),
    )

    assert get_pipeline_cfg_hash(training_metadata) == "cfg-hash-123"


def test_get_pipeline_cfg_hash_raises_when_hash_missing() -> None:
    """Raise PersistenceError when training metadata does not provide pipeline hash."""
    training_metadata = cast(
        TrainingMetadata,
        SimpleNamespace(config_fingerprint=SimpleNamespace(pipeline_cfg_hash="")),
    )

    with pytest.raises(PersistenceError, match="missing pipeline configuration hash"):
        get_pipeline_cfg_hash(training_metadata)


def test_get_training_conda_env_hash_returns_runtime_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    """Load runtime payload and return validated conda environment hash."""
    monkeypatch.setattr("ml.promotion.getters.get.load_json", lambda _: {"environment": {"conda_env_hash": "env-hash-1"}})
    monkeypatch.setattr(
        "ml.promotion.getters.get.validate_runtime_info",
        lambda _: SimpleNamespace(environment=SimpleNamespace(conda_env_hash="env-hash-1")),
    )

    result = get_training_conda_env_hash(Path("train_run"))

    assert result == "env-hash-1"


def test_get_training_conda_env_hash_raises_when_hash_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Raise PersistenceError when validated runtime metadata has empty conda hash."""
    monkeypatch.setattr("ml.promotion.getters.get.load_json", lambda _: {"environment": {"conda_env_hash": ""}})
    monkeypatch.setattr(
        "ml.promotion.getters.get.validate_runtime_info",
        lambda _: SimpleNamespace(environment=SimpleNamespace(conda_env_hash="")),
    )

    with pytest.raises(PersistenceError, match="missing conda environment hash"):
        get_training_conda_env_hash(Path("train_run"))


def test_get_runners_metadata_loads_each_stage_metadata_and_validates_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Load train/eval/explain metadata files and return validated stage objects in wrapper order."""
    train_run_dir = Path("runs") / "train-1"
    eval_run_dir = Path("runs") / "eval-1"
    explain_run_dir = Path("runs") / "explain-1"

    calls: list[str] = []

    def _load_json(path: Path) -> dict[str, str]:
        calls.append(f"load:{path}")
        if path == train_run_dir / "metadata.json":
            return {"stage": "training"}
        if path == eval_run_dir / "metadata.json":
            return {"stage": "evaluation"}
        if path == explain_run_dir / "metadata.json":
            return {"stage": "explainability"}
        raise AssertionError(f"Unexpected path: {path}")

    monkeypatch.setattr("ml.promotion.getters.get.load_json", _load_json)
    def _validate_training(payload: dict[str, Any]) -> str:
        calls.append(f"validate_training:{payload['stage']}")
        return "training-validated"

    def _validate_evaluation(payload: dict[str, Any]) -> str:
        calls.append(f"validate_evaluation:{payload['stage']}")
        return "evaluation-validated"

    def _validate_explainability(payload: dict[str, Any]) -> str:
        calls.append(f"validate_explainability:{payload['stage']}")
        return "explainability-validated"

    monkeypatch.setattr("ml.promotion.getters.get.validate_training_metadata", _validate_training)
    monkeypatch.setattr("ml.promotion.getters.get.validate_evaluation_metadata", _validate_evaluation)
    monkeypatch.setattr("ml.promotion.getters.get.validate_explainability_metadata", _validate_explainability)

    result = get_runners_metadata(train_run_dir, eval_run_dir, explain_run_dir)

    assert result.training_metadata == "training-validated"
    assert result.evaluation_metadata == "evaluation-validated"
    assert result.explainability_metadata == "explainability-validated"
    assert calls == [
        f"load:{train_run_dir / 'metadata.json'}",
        "validate_training:training",
        f"load:{eval_run_dir / 'metadata.json'}",
        "validate_evaluation:evaluation",
        f"load:{explain_run_dir / 'metadata.json'}",
        "validate_explainability:explainability",
    ]


def test_get_runners_metadata_propagates_validation_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Propagate stage validation exceptions so promotion flow can fail fast with context."""
    train_run_dir = Path("runs") / "train-err"
    eval_run_dir = Path("runs") / "eval-err"
    explain_run_dir = Path("runs") / "explain-err"

    monkeypatch.setattr("ml.promotion.getters.get.load_json", lambda _path: {"stage": "training"})

    class _ValidationError(RuntimeError):
        pass

    monkeypatch.setattr(
        "ml.promotion.getters.get.validate_training_metadata",
        lambda _payload: (_ for _ in ()).throw(_ValidationError("bad training metadata")),
    )

    with pytest.raises(_ValidationError, match="bad training metadata"):
        get_runners_metadata(train_run_dir, eval_run_dir, explain_run_dir)
