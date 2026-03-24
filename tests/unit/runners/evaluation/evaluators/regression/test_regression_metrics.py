"""Unit tests for regression evaluation metric helpers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from ml.config.schemas.model_specs import TargetTransformConfig
from ml.exceptions import EvaluationError
from ml.runners.evaluation.constants.data_splits import DataSplits
from ml.runners.evaluation.evaluators.regression import metrics as module
from sklearn.pipeline import Pipeline

pytestmark = pytest.mark.unit


class _PredictPipeline:
    """Minimal pipeline stub exposing ``predict`` for regression tests."""

    def __init__(self, values: np.ndarray) -> None:
        """Store fixed prediction array returned for every call."""
        self._values = values

    def predict(self, _X: pd.DataFrame) -> np.ndarray:
        """Return fixed predictions used by test scenarios."""
        return self._values


def test_compute_metrics_returns_core_regression_statistics() -> None:
    """Compute MAE/MSE/RMSE/R2 and residual summary values."""
    y_true = pd.Series([1.0, 2.0, 3.0])
    y_pred = pd.Series([1.5, 2.5, 2.0])

    metrics = module.compute_metrics(y_true, y_pred)

    assert set(metrics.keys()) == {
        "mae",
        "mse",
        "rmse",
        "r2",
        "median_ae",
        "explained_variance",
        "residual_mean",
        "residual_std",
    }
    assert metrics["rmse"] == pytest.approx(np.sqrt(metrics["mse"]))


def test_evaluate_split_applies_inverse_transform_and_builds_predictions_df(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Apply inverse target transform before metrics and persist residual column."""
    pipeline = _PredictPipeline(np.array([0.0, 1.0]))
    X = pd.DataFrame({"x": [1.0, 2.0]})
    y = pd.Series([2.0, 3.0])

    monkeypatch.setattr(module, "ensure_1d_array", lambda arr: arr)
    monkeypatch.setattr(module, "inverse_transform_target", lambda arr, **kwargs: arr + 1.0)

    metrics, df_preds = module.evaluate_split(
        cast(Pipeline, pipeline),
        X,
        y,
        split_entity_keys=pd.Series(["r1", "r2"]),
        split_name="val",
        transform_cfg=TargetTransformConfig(enabled=False, type=None, lambda_value=None),
        entity_key="entity_key",
    )

    # After inverse transform (+1), predictions become [1.0, 2.0] for y_true [2.0, 3.0].
    assert metrics["mae"] == pytest.approx(1.0)
    assert metrics["rmse"] == pytest.approx(1.0)
    assert df_preds.columns.tolist() == ["entity_key", "split", "y_true", "y_pred", "residual"]
    assert df_preds["split"].tolist() == ["val", "val"]
    assert df_preds["y_pred"].tolist() == [1.0, 2.0]
    assert df_preds["residual"].tolist() == [1.0, 1.0]


def test_evaluate_model_aggregates_splits_and_prediction_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Evaluate train/val/test splits and return typed prediction artifacts."""
    X = pd.DataFrame({"entity_key": ["r1", "r2"], "f": [1.0, 2.0]})
    y = pd.Series([1.0, 2.0])
    splits = DataSplits(train=(X, y), val=(X, y), test=(X, y))

    monkeypatch.setattr(module, "get_entity_keys", lambda frame, entity_key: frame[entity_key].copy())

    def _eval_split(**kwargs: Any) -> tuple[dict[str, float], pd.DataFrame]:
        split_name = kwargs["split_name"]
        df = pd.DataFrame(
            {
                "entity_key": ["r1", "r2"],
                "split": [split_name, split_name],
                "y_true": [1.0, 2.0],
                "y_pred": [1.0, 2.0],
                "residual": [0.0, 0.0],
            }
        )
        return {"rmse": 0.0}, df

    monkeypatch.setattr(module, "evaluate_split", _eval_split)

    metrics, artifacts = module.evaluate_model(
        pipeline=cast(Pipeline, SimpleNamespace()),
        data_splits=splits,
        transform_cfg=TargetTransformConfig(enabled=False, type=None, lambda_value=None),
        entity_key="entity_key",
    )

    assert set(metrics.keys()) == {"train", "val", "test"}
    assert artifacts.train["split"].iloc[0] == "train"
    assert artifacts.val["split"].iloc[0] == "val"
    assert artifacts.test["split"].iloc[0] == "test"


def test_evaluate_model_wraps_prediction_artifact_type_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wrap ``PredictionArtifacts`` constructor type errors as ``EvaluationError``."""
    X = pd.DataFrame({"entity_key": ["r1"], "f": [1.0]})
    y = pd.Series([1.0])
    splits = DataSplits(train=(X, y), val=(X, y), test=(X, y))
    monkeypatch.setattr(module, "get_entity_keys", lambda frame, entity_key: frame[entity_key].copy())
    monkeypatch.setattr(
        module,
        "evaluate_split",
        lambda **kwargs: ({"rmse": 0.0}, pd.DataFrame({"entity_key": ["r1"], "split": [kwargs["split_name"]]})),
    )

    class _FailingArtifacts:
        """Prediction artifacts stub that raises ``TypeError`` on construction."""

        def __init__(self, **_kwargs: Any) -> None:
            """Raise error to exercise EvaluationError wrapping branch."""
            raise TypeError("bad artifacts")

    monkeypatch.setattr(module, "PredictionArtifacts", _FailingArtifacts)

    with pytest.raises(EvaluationError, match="Error constructing PredictionArtifacts"):
        module.evaluate_model(
            pipeline=cast(Pipeline, SimpleNamespace()),
            data_splits=splits,
            transform_cfg=TargetTransformConfig(enabled=False, type=None, lambda_value=None),
            entity_key="entity_key",
        )


def test_evaluate_model_drops_row_id_before_split_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drop ``row_id`` from feature matrices before invoking split evaluator."""
    X = pd.DataFrame({"row_id": ["r1", "r2"], "f": [1.0, 2.0]})
    y = pd.Series([1.0, 2.0])
    splits = DataSplits(train=(X, y), val=(X, y), test=(X, y))

    monkeypatch.setattr(module, "get_entity_keys", lambda frame, entity_key: frame[entity_key].copy())

    seen_columns: dict[str, list[str]] = {}

    def _eval_split(**kwargs: Any) -> tuple[dict[str, float], pd.DataFrame]:
        split_name = cast(str, kwargs["split_name"])
        split_frame = cast(pd.DataFrame, kwargs["X"])
        seen_columns[split_name] = split_frame.columns.tolist()
        return {"rmse": 0.0}, pd.DataFrame({"row_id": ["r1"], "split": [split_name]})

    monkeypatch.setattr(module, "evaluate_split", _eval_split)

    module.evaluate_model(
        pipeline=cast(Pipeline, SimpleNamespace()),
        data_splits=splits,
        transform_cfg=TargetTransformConfig(enabled=False, type=None, lambda_value=None),
        entity_key="row_id",
    )

    assert seen_columns == {"train": ["f"], "val": ["f"], "test": ["f"]}


def test_evaluate_model_wraps_prediction_artifact_runtime_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wrap non-``TypeError`` artifact-construction failures as ``EvaluationError``."""
    X = pd.DataFrame({"entity_key": ["r1"], "f": [1.0]})
    y = pd.Series([1.0])
    splits = DataSplits(train=(X, y), val=(X, y), test=(X, y))
    monkeypatch.setattr(module, "get_entity_keys", lambda frame, entity_key: frame[entity_key].copy())
    monkeypatch.setattr(
        module,
        "evaluate_split",
        lambda **kwargs: ({"rmse": 0.0}, pd.DataFrame({"entity_key": ["r1"], "split": [kwargs["split_name"]]})),
    )

    class _FailingArtifacts:
        """Prediction artifacts stub that raises ``RuntimeError`` on construction."""

        def __init__(self, **_kwargs: Any) -> None:
            """Raise error to exercise generic wrapper branch."""
            raise RuntimeError("bad artifacts")

    monkeypatch.setattr(module, "PredictionArtifacts", _FailingArtifacts)

    with pytest.raises(EvaluationError, match="Error constructing PredictionArtifacts"):
        module.evaluate_model(
            pipeline=cast(Pipeline, SimpleNamespace()),
            data_splits=splits,
            transform_cfg=TargetTransformConfig(enabled=False, type=None, lambda_value=None),
            entity_key="entity_key",
        )
