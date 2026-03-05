"""Unit tests for promotion input getter utility helpers."""

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from ml.exceptions import PersistenceError, UserError
from ml.metadata.schemas.runners.training import TrainingMetadata
from ml.promotion.getters.get import (
    extract_thresholds,
    get_pipeline_cfg_hash,
    get_training_conda_env_hash,
)

pytestmark = pytest.mark.unit


def test_extract_thresholds_returns_problem_segment_threshold_mapping() -> None:
    """Select the exact threshold dictionary for the requested problem and segment."""
    thresholds = {
        "cancellation": {
            "city_hotel": {"promotion_metrics": {"sets": ["val"]}},
        }
    }

    result = extract_thresholds(thresholds, "cancellation", "city_hotel")

    assert result == {"promotion_metrics": {"sets": ["val"]}}


def test_extract_thresholds_raises_when_problem_segment_not_found() -> None:
    """Raise UserError when no threshold config exists for requested problem/segment."""
    thresholds = {"cancellation": {}}

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
