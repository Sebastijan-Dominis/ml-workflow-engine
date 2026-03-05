"""Unit tests for the compute_metrics function in ml.runners.training.utils.metrics.compute_metrics, which computes evaluation metrics for classification and regression models based on the model configuration and the provided training and validation data. The tests verify that the correct metrics are computed for classification tasks (including AUC and threshold metrics) and regression tasks (including RMSE, MAE, and R2), and that an error is raised for unsupported task types."""
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
    """Mock classification model with a method to get the best iteration."""
    def get_best_iteration(self) -> int:
        """Mock method to simulate getting the best iteration from a classification model."""
        return 5


class _ClassificationPipeline:
    """Mock classification pipeline with a predict_proba method that returns different probabilities based on the input size."""
    def predict_proba(self, X, ntree_end=None) -> np.ndarray:
        """Mock method to simulate predict_proba for a classification pipeline, returning different probabilities based on the input size.

        Args:
            X: The input data for which to predict probabilities. The method checks the length of X to determine which set of probabilities to return.
            ntree_end: An optional parameter that is not used in this mock implementation but is included to match the expected signature of a predict_proba method in a classification pipeline.

        Returns: A numpy array of predicted probabilities for the positive class, with different values based on the length of X.
        """
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
    """Mock regression pipeline with a predict method that returns different predictions based on the input size."""
    def predict(self, X) -> np.ndarray:
        """Mock method to simulate predict for a regression pipeline, returning different predictions based on the input size.

        Args:
            X: The input data for which to predict values. The method checks the length of X to determine which set of predictions to return.

        Returns:
            A numpy array of predicted values, with different values based on the length of X.
        """
        if len(X) == 3:
            return np.array([100.0, 110.0, 90.0])
        return np.array([120.0, 80.0])


def _base_cfg(task_type: str, subtype: str | None = None) -> SimpleNamespace:
    """Helper function to create a base configuration object for testing compute_metrics, with the task type and subtype specified.

    Args:
        task_type (str): The type of the task (e.g., "classification", "regression").
        subtype (str | None): The subtype of the task (e.g., "binary", "multiclass").

    Returns:
        SimpleNamespace: A base configuration object for testing compute_metrics.
    """
    return SimpleNamespace(
        task=SimpleNamespace(type=task_type, subtype=subtype),
        target=SimpleNamespace(
            transform=SimpleNamespace(enabled=False, type="log1p", lambda_value=None)
        ),
    )


def test_compute_metrics_classification_includes_auc_and_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that compute_metrics for a classification task includes AUC metrics and computes the best F1 threshold, with the get_best_f1_threshold function mocked to return a specific threshold value and F1 score.

    Args:
        monkeypatch (pytest.MonkeyPatch): The pytest monkeypatch fixture used to mock the get_best_f1_threshold function.
    """
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


def test_compute_metrics_regression_returns_expected_metric_keys() -> None:
    """Test that compute_metrics for a regression task returns a dictionary containing the expected metric keys (train_rmse, val_rmse, train_mae, val_mae, train_r2, val_r2) and that the computed metric values are approximately correct based on the mock regression pipeline's predictions."""
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
    """Test that compute_metrics raises a UserError when an unsupported task type (e.g., "ranking") is specified in the model configuration."""
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
