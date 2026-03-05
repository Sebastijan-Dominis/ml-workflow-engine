"""Unit tests for training configuration schema."""

import pytest
from ml.config.schemas.train_cfg import TrainConfig

pytestmark = pytest.mark.unit


def test_train_config_defaults_to_cpu_and_runtime_defaults() -> None:
    """Test that the TrainConfig schema defaults to CPU hardware and reasonable runtime defaults."""
    cfg = TrainConfig.model_validate({"iterations": 200})

    assert cfg.hardware.task_type == "CPU"
    assert cfg.early_stopping_rounds == 0
    assert cfg.snapshot_interval_seconds == 600


def test_train_config_accepts_explicit_model_and_ensemble_values() -> None:
    """Test that the TrainConfig schema correctly accepts and validates explicit model and ensemble hyperparameters."""
    cfg = TrainConfig.model_validate(
        {
            "iterations": 150,
            "model": {
                "depth": 6,
                "learning_rate": 0.05,
            },
            "ensemble": {
                "bagging_temperature": 0.8,
                "colsample_bylevel": 0.7,
            },
            "hardware": {
                "task_type": "gpu",
                "devices": [0],
            },
            "early_stopping_rounds": 25,
            "snapshot_interval_seconds": 120,
        }
    )

    assert cfg.model.depth == 6
    assert cfg.model.learning_rate == pytest.approx(0.05)
    assert cfg.ensemble.bagging_temperature == pytest.approx(0.8)
    assert cfg.ensemble.colsample_bylevel == pytest.approx(0.7)
    assert cfg.hardware.task_type == "GPU"
    assert cfg.hardware.devices == [0]
    assert cfg.early_stopping_rounds == 25
    assert cfg.snapshot_interval_seconds == 120
