"""Unit tests for evaluation/explainability artifacts validation wrappers."""

from __future__ import annotations

import pytest
from ml.exceptions import RuntimeMLError
from ml.metadata.schemas.runners.evaluation import EvaluationArtifacts
from ml.metadata.schemas.runners.explainability import ExplainabilityArtifacts
from ml.modeling.validation.artifacts import (
    validate_evaluation_artifacts,
    validate_explainability_artifacts,
)

pytestmark = pytest.mark.unit


def test_validate_evaluation_artifacts_returns_typed_model_for_valid_payload() -> None:
    """Construct an ``EvaluationArtifacts`` model when all required fields are present."""
    payload = {
        "model_hash": "model-hash-1",
        "model_path": "artifacts/model.cbm",
        "pipeline_path": "artifacts/pipeline.pkl",
        "pipeline_hash": "pipeline-hash-1",
        "train_predictions_path": "artifacts/pred_train.parquet",
        "val_predictions_path": "artifacts/pred_val.parquet",
        "test_predictions_path": "artifacts/pred_test.parquet",
        "train_predictions_hash": "pred-hash-train",
        "val_predictions_hash": "pred-hash-val",
        "test_predictions_hash": "pred-hash-test",
        "metrics_path": "artifacts/metrics.json",
        "metrics_hash": "metrics-hash-1",
    }

    result = validate_evaluation_artifacts(payload)

    assert isinstance(result, EvaluationArtifacts)
    assert result.metrics_hash == "metrics-hash-1"
    assert result.pipeline_hash == "pipeline-hash-1"


def test_validate_evaluation_artifacts_wraps_schema_errors_as_runtime_ml_error() -> None:
    """Wrap invalid evaluation artifact payloads as ``RuntimeMLError`` for callers."""
    payload = {
        "model_hash": "model-hash-1",
        "model_path": "artifacts/model.cbm",
        "train_predictions_path": "artifacts/pred_train.parquet",
        "val_predictions_path": "artifacts/pred_val.parquet",
        "test_predictions_path": "artifacts/pred_test.parquet",
        "train_predictions_hash": "pred-hash-train",
        "val_predictions_hash": "pred-hash-val",
        "test_predictions_hash": "pred-hash-test",
        "metrics_path": "artifacts/metrics.json",
        # metrics_hash intentionally missing
    }

    with pytest.raises(RuntimeMLError, match="Failed to construct evaluation artifacts model"):
        validate_evaluation_artifacts(payload)


def test_validate_explainability_artifacts_returns_typed_model_with_defaults() -> None:
    """Use schema defaults for optional explainability artifact fields when omitted."""
    payload = {
        "model_hash": "model-hash-2",
        "model_path": "artifacts/model.cbm",
    }

    result = validate_explainability_artifacts(payload)

    assert isinstance(result, ExplainabilityArtifacts)
    assert result.top_k_feature_importances_path == ""
    assert result.top_k_shap_importances_hash == ""


def test_validate_explainability_artifacts_wraps_schema_errors_as_runtime_ml_error() -> None:
    """Wrap invalid explainability artifact payloads as ``RuntimeMLError`` consistently."""
    payload = {
        "model_path": "artifacts/model.cbm",
        # model_hash intentionally missing
    }

    with pytest.raises(RuntimeMLError, match="Failed to construct explainability artifacts model"):
        validate_explainability_artifacts(payload)
