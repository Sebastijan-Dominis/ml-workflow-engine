"""Unit tests for training/evaluation metrics validation wrappers."""

from __future__ import annotations

import pytest
from ml.exceptions import RuntimeMLError
from ml.modeling.models.metrics import EvaluationMetrics, TrainingMetrics
from ml.modeling.validation.metrics import (
    validate_evaluation_metrics,
    validate_training_metrics,
)

pytestmark = pytest.mark.unit


def test_validate_training_metrics_returns_typed_model_for_valid_payload() -> None:
    """Return a ``TrainingMetrics`` model when payload satisfies schema constraints."""
    payload = {
        "task_type": "classification",
        "algorithm": "catboost",
        "metrics": {
            "accuracy": 0.89,
            "f1": 0.84,
        },
    }

    result = validate_training_metrics(payload)

    assert isinstance(result, TrainingMetrics)
    assert result.task_type == "classification"
    assert result.algorithm == "catboost"
    assert result.metrics["accuracy"] == pytest.approx(0.89)


def test_validate_training_metrics_wraps_schema_errors_as_runtime_ml_error() -> None:
    """Wrap invalid training metric payloads as ``RuntimeMLError`` for callers."""
    payload = {
        "task_type": "classification",
        "algorithm": "catboost",
        "metrics": "not-a-dict",
    }

    with pytest.raises(RuntimeMLError, match="Error validating training metrics"):
        validate_training_metrics(payload)


def test_validate_evaluation_metrics_returns_typed_model_for_valid_payload() -> None:
    """Return an ``EvaluationMetrics`` model for valid split-wise metrics payloads."""
    payload = {
        "task_type": "classification",
        "algorithm": "catboost",
        "train": {"accuracy": 0.91},
        "val": {"accuracy": 0.88},
        "test": {"accuracy": 0.87},
    }

    result = validate_evaluation_metrics(payload)

    assert isinstance(result, EvaluationMetrics)
    assert result.val["accuracy"] == pytest.approx(0.88)


def test_validate_evaluation_metrics_wraps_schema_errors_as_runtime_ml_error() -> None:
    """Wrap invalid evaluation metric payloads as ``RuntimeMLError`` for uniform handling."""
    payload = {
        "task_type": "classification",
        "algorithm": "catboost",
        "train": {"accuracy": 0.91},
        "val": {"accuracy": 0.88},
        # test split missing on purpose
    }

    with pytest.raises(RuntimeMLError, match="Error validating evaluation metrics"):
        validate_evaluation_metrics(payload)
