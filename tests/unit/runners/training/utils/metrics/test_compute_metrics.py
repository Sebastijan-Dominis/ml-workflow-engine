from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from ml.exceptions import UserError
from ml.runners.training.utils.metrics.compute_metrics import compute_metrics

pytestmark = pytest.mark.unit


class _ClassificationModel:
    def get_best_iteration(self) -> int:
        return 5


class _ClassificationPipeline:
    def predict_proba(self, X, ntree_end=None):
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
    def predict(self, X):
        if len(X) == 3:
            return np.array([100.0, 110.0, 90.0])
        return np.array([120.0, 80.0])


def _base_cfg(task_type: str, subtype: str | None = None):
    return SimpleNamespace(
        task=SimpleNamespace(type=task_type, subtype=subtype),
        target=SimpleNamespace(
            transform=SimpleNamespace(enabled=False, type="log1p", lambda_value=None)
        ),
    )


def test_compute_metrics_classification_includes_auc_and_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "ml.runners.training.utils.metrics.compute_metrics.get_best_f1_threshold",
        lambda pipeline, X, y: (0.42, 0.77),
    )

    metrics = compute_metrics(
        model=_ClassificationModel(),
        pipeline=_ClassificationPipeline(),
        model_cfg=_base_cfg("classification", "binary"),
        X_train=pd.DataFrame({"x": [1, 2, 3, 4]}),
        y_train=pd.Series([0, 1, 0, 1]),
        X_val=pd.DataFrame({"x": [10, 11, 12, 13]}),
        y_val=pd.Series([0, 1, 1, 0]),
    )

    assert metrics["best_iteration"] == 5
    assert "train_auc" in metrics and "val_auc" in metrics
    assert metrics["threshold"] == {"value": 0.42, "f1": 0.77}


def test_compute_metrics_regression_returns_expected_metric_keys() -> None:
    metrics = compute_metrics(
        model=object(),
        pipeline=_RegressionPipeline(),
        model_cfg=_base_cfg("regression"),
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
    with pytest.raises(UserError, match="Task type ranking not supported"):
        compute_metrics(
            model=object(),
            pipeline=object(),
            model_cfg=_base_cfg("ranking"),
            X_train=pd.DataFrame({"x": [1]}),
            y_train=pd.Series([1]),
            X_val=pd.DataFrame({"x": [2]}),
            y_val=pd.Series([1]),
        )
