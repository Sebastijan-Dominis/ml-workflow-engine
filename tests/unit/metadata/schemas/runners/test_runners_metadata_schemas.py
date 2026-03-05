"""Unit tests for runner metadata schema models.

These tests verify that training/evaluation/explainability metadata schemas
accept valid payloads and enforce required identity/artifact fields.
"""

import pytest
from ml.metadata.schemas.runners.evaluation import EvaluationMetadata
from ml.metadata.schemas.runners.explainability import ExplainabilityMetadata
from ml.metadata.schemas.runners.training import TrainingMetadata
from pydantic import ValidationError

pytestmark = pytest.mark.unit


def _lineage_payload() -> dict:
    """Return a valid experiment lineage payload for runner metadata tests."""

    return {
        "feature_lineage": [
            {
                "name": "booking_context_features",
                "version": "v1",
                "snapshot_id": "20260305T120000_abcd",
                "file_hash": "file-hash",
                "in_memory_hash": "mem-hash",
                "feature_schema_hash": "schema-hash",
                "operator_hash": "op-hash",
                "feature_type": "tabular",
            }
        ],
        "target_column": "is_canceled",
        "problem": "cancellation",
        "segment": "city_hotel_online_ta",
        "model_version": "v1",
    }


def _config_fingerprint_payload() -> dict:
    """Return a valid configuration fingerprint payload."""

    return {
        "config_hash": "cfg-hash",
        "pipeline_cfg_hash": "pipeline-cfg-hash",
    }


def _base_artifacts_payload() -> dict:
    """Return a valid base artifacts payload used across runner schemas."""

    return {
        "model_hash": "model-hash",
        "model_path": "artifacts/model.cbm",
        "pipeline_path": "artifacts/pipeline.pkl",
        "pipeline_hash": "pipeline-hash",
    }


def test_training_metadata_accepts_valid_payload() -> None:
    """Ensure TrainingMetadata accepts a fully valid payload."""

    payload = {
        "run_identity": {
            "stage": "training",
            "train_run_id": "train-run-001",
            "snapshot_id": "20260305T120000_abcd",
            "status": "success",
        },
        "lineage": _lineage_payload(),
        "config_fingerprint": _config_fingerprint_payload(),
        "artifacts": _base_artifacts_payload(),
    }

    result = TrainingMetadata.model_validate(payload)

    assert result.run_identity.stage == "training"
    assert result.artifacts.model_hash == "model-hash"


def test_training_metadata_rejects_wrong_stage_literal() -> None:
    """Ensure TrainingMetadata enforces the `training` stage literal."""

    payload = {
        "run_identity": {
            "stage": "evaluation",
            "train_run_id": "train-run-001",
            "snapshot_id": "20260305T120000_abcd",
            "status": "success",
        },
        "lineage": _lineage_payload(),
        "config_fingerprint": _config_fingerprint_payload(),
        "artifacts": _base_artifacts_payload(),
    }

    with pytest.raises(ValidationError, match="training"):
        TrainingMetadata.model_validate(payload)


def test_evaluation_metadata_accepts_valid_payload() -> None:
    """Ensure EvaluationMetadata accepts valid run identity and artifacts fields."""

    payload = {
        "run_identity": {
            "stage": "evaluation",
            "train_run_id": "train-run-001",
            "snapshot_id": "20260305T120000_abcd",
            "status": "success",
            "eval_run_id": "eval-run-001",
        },
        "lineage": _lineage_payload(),
        "config_fingerprint": _config_fingerprint_payload(),
        "artifacts": {
            **_base_artifacts_payload(),
            "train_predictions_path": "preds/train.parquet",
            "val_predictions_path": "preds/val.parquet",
            "test_predictions_path": "preds/test.parquet",
            "train_predictions_hash": "train-hash",
            "val_predictions_hash": "val-hash",
            "test_predictions_hash": "test-hash",
            "metrics_path": "metrics/eval.json",
            "metrics_hash": "metrics-hash",
        },
    }

    result = EvaluationMetadata.model_validate(payload)

    assert result.run_identity.eval_run_id == "eval-run-001"
    assert result.artifacts.metrics_hash == "metrics-hash"


def test_evaluation_metadata_rejects_missing_eval_run_id() -> None:
    """Ensure EvaluationMetadata requires `eval_run_id` in run identity."""

    payload = {
        "run_identity": {
            "stage": "evaluation",
            "train_run_id": "train-run-001",
            "snapshot_id": "20260305T120000_abcd",
            "status": "success",
        },
        "lineage": _lineage_payload(),
        "config_fingerprint": _config_fingerprint_payload(),
        "artifacts": {
            **_base_artifacts_payload(),
            "train_predictions_path": "preds/train.parquet",
            "val_predictions_path": "preds/val.parquet",
            "test_predictions_path": "preds/test.parquet",
            "train_predictions_hash": "train-hash",
            "val_predictions_hash": "val-hash",
            "test_predictions_hash": "test-hash",
            "metrics_path": "metrics/eval.json",
            "metrics_hash": "metrics-hash",
        },
    }

    with pytest.raises(ValidationError, match="eval_run_id"):
        EvaluationMetadata.model_validate(payload)


def test_explainability_metadata_accepts_valid_payload() -> None:
    """Ensure ExplainabilityMetadata accepts a valid payload and top_k value."""

    payload = {
        "run_identity": {
            "train_run_id": "train-run-001",
            "snapshot_id": "20260305T120000_abcd",
            "status": "success",
            "explain_run_id": "explain-run-001",
        },
        "lineage": _lineage_payload(),
        "config_fingerprint": _config_fingerprint_payload(),
        "artifacts": {
            **_base_artifacts_payload(),
            "top_k_feature_importances_path": "explain/fi.parquet",
            "top_k_feature_importances_hash": "fi-hash",
            "top_k_shap_importances_path": "explain/shap.parquet",
            "top_k_shap_importances_hash": "shap-hash",
        },
        "top_k": 20,
    }

    result = ExplainabilityMetadata.model_validate(payload)

    assert result.run_identity.stage == "explainability"
    assert result.top_k == 20
