import json
import types
from pathlib import Path
from typing import Any, cast

import pytest
from ml.config.schemas.model_cfg import TrainModelConfig
from ml.exceptions import PersistenceError
from ml.runners.shared.persistence.save_metrics import save_metrics


def _make_dummy_model_cfg(task_type: str = "classification", algorithm_value: str = "catboost") -> TrainModelConfig:
    # Build a minimal object exposing the attributes used by save_metrics.
    dummy = types.SimpleNamespace(task=types.SimpleNamespace(type=task_type), algorithm=types.SimpleNamespace(value=algorithm_value))
    return cast(TrainModelConfig, dummy)


def test_save_metrics_writes_training_metrics(tmp_path: Path) -> None:
    model_cfg = _make_dummy_model_cfg()
    experiment_dir = tmp_path / "exp"
    metrics = {"accuracy": 0.9}

    path = save_metrics(metrics, model_cfg=model_cfg, target_run_id="run1", experiment_dir=experiment_dir, stage="training")

    metrics_file = Path(path)
    assert metrics_file.exists()

    data = json.loads(metrics_file.read_text(encoding="utf-8"))
    assert data["metrics"]["accuracy"] == 0.9
    assert data["task_type"] == "classification"
    assert data["algorithm"] == "catboost"


def test_save_metrics_raises_for_invalid_stage(tmp_path: Path) -> None:
    model_cfg = _make_dummy_model_cfg()

    with pytest.raises(PersistenceError):
        # pass an invalid stage at runtime; cast to Any to satisfy static typing
        save_metrics({"m": 1.0}, model_cfg=model_cfg, target_run_id="r", experiment_dir=tmp_path, stage=cast(Any, "invalid"))
