"""Unit tests for classification evaluation metric helpers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from ml.config.schemas.model_cfg import TrainModelConfig
from ml.exceptions import EvaluationError, PipelineContractError, UserError
from ml.runners.evaluation.constants.data_splits import DataSplits
from ml.runners.evaluation.evaluators.classification import metrics as module
from sklearn.pipeline import Pipeline

pytestmark = pytest.mark.unit


class _ProbPipeline:
    """Minimal pipeline stub exposing ``predict_proba`` for classification tests."""

    def __init__(self, probs: np.ndarray) -> None:
        """Store fixed probability output returned for every prediction call."""
        self._probs = probs

    def predict_proba(self, _X: pd.DataFrame) -> np.ndarray:
        """Return fixed probability matrix used by test scenarios."""
        return self._probs


def test_expected_calibration_error_returns_float_for_sparse_bins() -> None:
    """Return stable float ECE even when many bins have no assigned samples."""
    y_true = pd.Series([0, 1])
    y_prob = pd.Series([0.01, 0.99])

    ece = module.expected_calibration_error(y_true, y_prob, n_bins=20)

    assert isinstance(ece, float)
    assert 0.0 <= ece <= 1.0


def test_compute_metrics_without_probabilities_sets_prob_metrics_to_nan() -> None:
    """Populate threshold metrics while leaving probability-only metrics as NaN."""
    y_true = pd.Series([0, 1, 1, 0])
    y_pred = pd.Series([0, 1, 0, 0])

    metrics = module.compute_metrics(y_true, y_pred, y_prob=None, threshold=0.4)

    assert metrics["threshold"] == 0.4
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert np.isnan(metrics["roc_auc"])
    assert np.isnan(metrics["pr_auc"])
    assert np.isnan(metrics["log_loss"])
    assert np.isnan(metrics["brier_score"])
    assert np.isnan(metrics["ece"])


def test_compute_metrics_with_probabilities_computes_probability_metrics() -> None:
    """Compute ROC/PR/log-loss/Brier/ECE when probability scores are provided."""
    y_true = pd.Series([0, 1, 1, 0])
    y_pred = pd.Series([0, 1, 1, 0])
    y_prob = pd.Series([0.1, 0.9, 0.8, 0.2])

    metrics = module.compute_metrics(y_true, y_pred, y_prob=y_prob, threshold=0.5)

    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["pr_auc"] == pytest.approx(1.0)
    assert metrics["log_loss"] >= 0.0
    assert metrics["brier_score"] >= 0.0
    assert metrics["ece"] >= 0.0


def test_compute_metrics_sets_specificity_nan_when_confusion_matrix_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Set specificity to NaN when confusion matrix cannot be derived."""
    monkeypatch.setattr(module, "confusion_matrix", lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("bad")))

    metrics = module.compute_metrics(pd.Series([0, 1]), pd.Series([0, 1]), y_prob=None)

    assert np.isnan(metrics["specificity"])


def test_compute_metrics_sets_specificity_zero_when_no_negative_denominator() -> None:
    """Set specificity to 0.0 when confusion-matrix TN+FP denominator is zero."""
    y_true = pd.Series([1, 1, 1, 1])
    y_pred = pd.Series([0, 0, 0, 0])

    metrics = module.compute_metrics(y_true, y_pred, y_prob=None)

    assert metrics["specificity"] == 0.0


def test_compute_metrics_sets_probability_metrics_nan_when_metric_functions_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Set probability-based metrics to NaN when underlying metric calls raise ``ValueError``."""

    def _raise_value_error(*_args: Any, **_kwargs: Any) -> float:
        raise ValueError("undefined")

    monkeypatch.setattr(module, "roc_auc_score", _raise_value_error)
    monkeypatch.setattr(module, "average_precision_score", _raise_value_error)
    monkeypatch.setattr(module, "log_loss", _raise_value_error)
    monkeypatch.setattr(module, "brier_score_loss", _raise_value_error)

    y_true = pd.Series([0, 1, 0, 1])
    y_pred = pd.Series([0, 1, 1, 0])
    y_prob = pd.Series([0.1, 0.9, 0.8, 0.2])

    metrics = module.compute_metrics(y_true, y_pred, y_prob=y_prob, threshold=0.5)

    assert np.isnan(metrics["roc_auc"])
    assert np.isnan(metrics["pr_auc"])
    assert np.isnan(metrics["log_loss"])
    assert np.isnan(metrics["brier_score"])
    assert isinstance(metrics["ece"], float)


def test_evaluate_split_returns_predictions_dataframe_for_valid_binary_probabilities() -> None:
    """Build split predictions output with thresholded labels for binary probabilities."""

    class _ValidProbPipeline:
        """Pipeline stub returning two-class probabilities for each input row."""

        def predict_proba(self, _X: pd.DataFrame) -> np.ndarray:
            """Return deterministic positive-class probabilities for thresholding."""
            return np.array([[0.8, 0.2], [0.1, 0.9]], dtype=float)

    pipeline = _ValidProbPipeline()
    X = pd.DataFrame({"x": [1, 2]})
    y = pd.Series([0, 1])

    metrics, df_preds = module.evaluate_split(
        cast(Pipeline, pipeline),
        X,
        y,
        split_row_ids=pd.Series(["r1", "r2"]),
        split_name="val",
        best_threshold=0.5,
    )

    assert "accuracy" in metrics
    assert df_preds.columns.tolist() == ["row_id", "split", "y_true", "y_pred", "y_proba"]
    assert df_preds["split"].tolist() == ["val", "val"]
    assert df_preds["y_pred"].tolist() == [0, 1]


def test_evaluate_split_raises_when_predict_proba_is_not_binary_matrix() -> None:
    """Raise ``PipelineContractError`` when probability output lacks positive class column."""
    pipeline = _ProbPipeline(np.array([[0.7], [0.2]], dtype=float))
    X = pd.DataFrame({"x": [1, 2]})
    y = pd.Series([0, 1])

    with pytest.raises(PipelineContractError, match="at least 2 columns"):
        module.evaluate_split(
            cast(Pipeline, pipeline),
            X,
            y,
            split_row_ids=pd.Series(["r1", "r2"]),
            split_name="train",
            best_threshold=0.5,
        )


def test_evaluate_model_binary_aggregates_split_metrics_and_predictions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Evaluate train/val/test splits for binary subtype and return artifacts."""
    X = pd.DataFrame({"row_id": ["r1", "r2"], "f": [1.0, 2.0]})
    y = pd.Series([0, 1])
    splits = DataSplits(train=(X, y), val=(X, y), test=(X, y))

    monkeypatch.setattr(module, "get_row_ids", lambda frame: frame["row_id"].copy())

    def _eval_split(**kwargs: Any) -> tuple[dict[str, float], pd.DataFrame]:
        split_name = kwargs["split_name"]
        df = pd.DataFrame(
            {
                "row_id": ["r1", "r2"],
                "split": [split_name, split_name],
                "y_true": [0, 1],
                "y_pred": [0, 1],
                "y_proba": [0.1, 0.9],
            }
        )
        return {"accuracy": 1.0}, df

    monkeypatch.setattr(module, "evaluate_split", _eval_split)

    model_cfg = cast(TrainModelConfig, SimpleNamespace(task=SimpleNamespace(type="classification", subtype="binary")))

    metrics, artifacts = module.evaluate_model(
        model_cfg,
        pipeline=cast(Pipeline, SimpleNamespace()),
        data_splits=splits,
        best_threshold=0.5,
    )

    assert set(metrics.keys()) == {"train", "val", "test"}
    assert artifacts.train["split"].iloc[0] == "train"
    assert artifacts.val["split"].iloc[0] == "val"
    assert artifacts.test["split"].iloc[0] == "test"


def test_evaluate_model_binary_drops_row_id_before_split_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drop ``row_id`` column from split features before invoking split evaluator."""
    X = pd.DataFrame({"row_id": ["r1", "r2"], "f": [1.0, 2.0]})
    y = pd.Series([0, 1])
    splits = DataSplits(train=(X, y), val=(X, y), test=(X, y))

    monkeypatch.setattr(module, "get_row_ids", lambda frame: frame["row_id"].copy())

    seen_columns: dict[str, list[str]] = {}

    def _eval_split(**kwargs: Any) -> tuple[dict[str, float], pd.DataFrame]:
        split_name = cast(str, kwargs["split_name"])
        split_frame = cast(pd.DataFrame, kwargs["X"])
        seen_columns[split_name] = split_frame.columns.tolist()
        return {"accuracy": 1.0}, pd.DataFrame({"row_id": ["r1"], "split": [split_name]})

    monkeypatch.setattr(module, "evaluate_split", _eval_split)

    model_cfg = cast(TrainModelConfig, SimpleNamespace(task=SimpleNamespace(type="classification", subtype="binary")))

    module.evaluate_model(
        model_cfg,
        pipeline=cast(Pipeline, SimpleNamespace()),
        data_splits=splits,
        best_threshold=0.5,
    )

    assert seen_columns == {"train": ["f"], "val": ["f"], "test": ["f"]}


def test_evaluate_model_raises_for_multiclass_subtype() -> None:
    """Raise ``UserError`` for multiclass subtype until implementation is provided."""
    X = pd.DataFrame({"row_id": ["r1"], "f": [1.0]})
    y = pd.Series([1])
    splits = DataSplits(train=(X, y), val=(X, y), test=(X, y))

    model_cfg = cast(TrainModelConfig, SimpleNamespace(task=SimpleNamespace(type="classification", subtype="multiclass")))

    with pytest.raises(UserError, match="not yet implemented"):
        module.evaluate_model(model_cfg, pipeline=cast(Pipeline, SimpleNamespace()), data_splits=splits, best_threshold=0.5)


def test_evaluate_model_raises_for_unsupported_subtype() -> None:
    """Raise ``PipelineContractError`` when classification subtype is unsupported."""
    X = pd.DataFrame({"row_id": ["r1"], "f": [1.0]})
    y = pd.Series([1])
    splits = DataSplits(train=(X, y), val=(X, y), test=(X, y))

    model_cfg = cast(TrainModelConfig, SimpleNamespace(task=SimpleNamespace(type="classification", subtype="ordinal")))

    with pytest.raises(PipelineContractError, match="Unsupported task subtype"):
        module.evaluate_model(model_cfg, pipeline=cast(Pipeline, SimpleNamespace()), data_splits=splits, best_threshold=0.5)


def test_evaluate_model_wraps_prediction_artifact_construction_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wrap prediction artifact construction failures as ``EvaluationError``."""
    X = pd.DataFrame({"row_id": ["r1"], "f": [1.0]})
    y = pd.Series([0])
    splits = DataSplits(train=(X, y), val=(X, y), test=(X, y))

    monkeypatch.setattr(module, "get_row_ids", lambda frame: frame["row_id"].copy())
    monkeypatch.setattr(
        module,
        "evaluate_split",
        lambda **kwargs: ({"accuracy": 1.0}, pd.DataFrame({"row_id": ["r1"], "split": [kwargs["split_name"]]})),
    )

    class _FailingArtifacts:
        """Prediction artifacts stub that always raises at construction."""

        def __init__(self, **_kwargs: Any) -> None:
            """Raise generic exception to exercise wrapper branch."""
            raise RuntimeError("bad artifacts")

    monkeypatch.setattr(module, "PredictionArtifacts", _FailingArtifacts)

    model_cfg = cast(TrainModelConfig, SimpleNamespace(task=SimpleNamespace(type="classification", subtype="binary")))

    with pytest.raises(EvaluationError, match="Failed to construct PredictionArtifacts"):
        module.evaluate_model(
            model_cfg,
            pipeline=cast(Pipeline, SimpleNamespace()),
            data_splits=splits,
            best_threshold=0.5,
        )
