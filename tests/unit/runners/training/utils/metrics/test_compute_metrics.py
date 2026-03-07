"""Unit tests for training metric computation."""
from types import SimpleNamespace
from typing import cast

import numpy as np
import pandas as pd
import pytest
from ml.config.schemas.model_cfg import TrainModelConfig
from ml.exceptions import UserError
from ml.runners.training.utils.metrics.compute_metrics import compute_metrics
from sklearn.pipeline import Pipeline

pytestmark = pytest.mark.unit


class _ClassificationModel:
    """Minimal classification model stub for tests."""
    def get_best_iteration(self) -> int:
        """Return a fixed best iteration for deterministic assertions."""
        return 5


class _ClassificationPipeline:
    """Classification pipeline stub with deterministic probabilities."""
    def predict_proba(self, X, ntree_end=None) -> np.ndarray:
        """Return deterministic probabilities keyed by input length."""
        if len(X) == 4:
            return np.array(
                [
                    [0.90, 0.10],
                    [0.20, 0.80],
                    [0.85, 0.15],
                    [0.10, 0.90],
                ]
            )
        return np.array(
            [
                [0.80, 0.20],
                [0.10, 0.90],
                [0.30, 0.70],
                [0.95, 0.05],
            ]
        )


class _RegressionPipeline:
    """Regression pipeline stub with deterministic predictions."""
    def predict(self, X) -> np.ndarray:
        """Return deterministic regression predictions keyed by input length."""
        if len(X) == 3:
            return np.array([100.0, 110.0, 90.0])
        return np.array([120.0, 80.0])


class _ForecastModel:
    """Forecasting model stub with deterministic predictions."""

    def predict(self, X) -> np.ndarray:
        """Return deterministic forecasting predictions keyed by input length."""
        if len(X) == 3:
            return np.array([10.0, 20.0, 30.0])
        return np.array([40.0, 50.0])


def _base_cfg(task_type: str, subtype: str | None = None) -> SimpleNamespace:
    """Build a lightweight model configuration stub for metric tests."""
    return SimpleNamespace(
        task=SimpleNamespace(type=task_type, subtype=subtype),
        target=SimpleNamespace(
            transform=SimpleNamespace(enabled=False, type="log1p", lambda_value=None)
        ),
    )


def test_compute_metrics_classification_includes_auc_and_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify classification metrics include AUC outputs and threshold metadata."""
    monkeypatch.setattr(
        "ml.runners.training.utils.metrics.compute_metrics.get_best_f1_threshold",
        lambda pipeline, X, y: (0.42, 0.77),
    )

    metrics = compute_metrics(
        model=_ClassificationModel(),
        pipeline=cast(Pipeline, _ClassificationPipeline()),
        model_cfg=cast(TrainModelConfig, _base_cfg("classification", "binary")),
        X_train=pd.DataFrame({"x": [1, 2, 3, 4]}),
        y_train=pd.Series([0, 1, 0, 1]),
        X_val=pd.DataFrame({"x": [10, 11, 12, 13]}),
        y_val=pd.Series([0, 1, 1, 0]),
    )

    assert metrics["best_iteration"] == 5
    assert "train_auc" in metrics and "val_auc" in metrics
    assert metrics["threshold"] == {"value": 0.42, "f1": 0.77}


def test_compute_metrics_classification_without_threshold_support_omits_threshold() -> None:
    """Verify non-threshold tasks do not include threshold metadata."""
    metrics = compute_metrics(
        model=_ClassificationModel(),
        pipeline=cast(Pipeline, _ClassificationPipeline()),
        model_cfg=cast(TrainModelConfig, _base_cfg("classification", "custom_binary")),
        X_train=pd.DataFrame({"x": [1, 2, 3, 4]}),
        y_train=pd.Series([0, 1, 0, 1]),
        X_val=pd.DataFrame({"x": [10, 11, 12, 13]}),
        y_val=pd.Series([0, 1, 1, 0]),
    )

    assert "threshold" not in metrics


def test_compute_metrics_forecasting_returns_expected_metric_keys() -> None:
    """Verify forecasting branch computes expected error metrics."""
    metrics = compute_metrics(
        model=_ForecastModel(),
        pipeline=cast(Pipeline, object()),
        model_cfg=cast(TrainModelConfig, _base_cfg("forecasting")),
        X_train=pd.DataFrame({"x": [1, 2, 3]}),
        y_train=pd.Series([10.0, 20.0, 30.0]),
        X_val=pd.DataFrame({"x": [4, 5]}),
        y_val=pd.Series([40.0, 50.0]),
    )

    assert set(metrics.keys()) == {
        "train_rmse",
        "val_rmse",
        "train_mae",
        "val_mae",
    }
    assert metrics["train_rmse"] == pytest.approx(0.0)
    assert metrics["val_mae"] == pytest.approx(0.0)


def test_compute_metrics_regression_returns_expected_metric_keys() -> None:
    """Verify regression metrics include the expected key set and values."""
    metrics = compute_metrics(
        model=object(),
        pipeline=cast(Pipeline, _RegressionPipeline()),
        model_cfg=cast(TrainModelConfig, _base_cfg("regression")),
        X_train=pd.DataFrame({"x": [1, 2, 3]}),
        y_train=pd.Series([100.0, 110.0, 90.0]),
        X_val=pd.DataFrame({"x": [10, 11]}),
        y_val=pd.Series([120.0, 80.0]),
    )

    assert set(metrics.keys()) == {
        "train_rmse",
        "val_rmse",
        "train_mae",
        "val_mae",
        "train_r2",
        "val_r2",
    }
    assert metrics["train_rmse"] == pytest.approx(0.0)
    assert metrics["val_mae"] == pytest.approx(0.0)


def test_compute_metrics_raises_for_unsupported_task_type() -> None:
    """Verify unsupported task types raise `UserError`."""
    with pytest.raises(UserError, match="Task type ranking not supported"):
        compute_metrics(
            model=object(),
            pipeline=cast(Pipeline, object()),
            model_cfg=cast(TrainModelConfig, _base_cfg("ranking")),
            X_train=pd.DataFrame({"x": [1]}),
            y_train=pd.Series([1]),
            X_val=pd.DataFrame({"x": [2]}),
            y_val=pd.Series([1]),
        )
