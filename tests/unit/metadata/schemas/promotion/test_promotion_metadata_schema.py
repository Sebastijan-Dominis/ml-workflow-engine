"""Unit tests for promotion metadata schema models.

These tests validate stage-specific run identity requirements and key optional
fields for production and staging promotion metadata.
"""

import pytest
from ml.metadata.schemas.promotion.promote import (
    ProductionPromotionMetadata,
    StagingPromotionMetadata,
)
from pydantic import ValidationError

pytestmark = pytest.mark.unit


def _promotion_base_payload() -> dict:
    """Return a valid base payload shared by staging and production schemas."""

    return {
        "previous_run_identity": {
            "experiment_id": "exp-prev",
            "train_run_id": "train-prev",
            "eval_run_id": "eval-prev",
            "explain_run_id": "explain-prev",
            "promotion_id": "promotion-prev",
        },
        "metrics": {
            "task_type": "classification",
            "algorithm": "catboost",
            "metrics": {
                "train": {"f1": 0.8, "roc_auc": 0.85},
                "val": {"f1": 0.78, "roc_auc": 0.83},
                "test": {"f1": 0.77, "roc_auc": 0.82},
            }
        },
        "previous_production_metrics": None,
        "promotion_thresholds": {
            "promotion_metrics": {
                "sets": ["val", "test"],
                "metrics": ["f1", "roc_auc"],
                "directions": {"f1": "maximize", "roc_auc": "maximize"},
            },
            "thresholds": {
                "val": {"f1": 0.7, "roc_auc": 0.8},
                "test": {"f1": 0.69, "roc_auc": 0.79},
            },
            "lineage": {"created_by": "tests", "created_at": "2026-03-05T12:00:00"},
        },
        "promotion_thresholds_hash": "thresholds-hash",
        "context": {
            "git_commit": "abc123",
            "promotion_conda_env_hash": "prom-env-hash",
            "training_conda_env_hash": "train-env-hash",
            "timestamp": "2026-03-05T12:00:00",
        },
    }


def test_production_promotion_metadata_allows_none_previous_metrics() -> None:
    """Ensure ProductionPromotionMetadata accepts None previous production metrics."""

    payload = {
        **_promotion_base_payload(),
        "run_identity": {
            "stage": "production",
            "experiment_id": "exp-current",
            "train_run_id": "train-current",
            "eval_run_id": "eval-current",
            "explain_run_id": "explain-current",
            "promotion_id": "promotion-current",
        },
        "decision": {
            "promoted": True,
            "reason": "beats previous",
            "beats_previous": True,
        },
    }

    result = ProductionPromotionMetadata.model_validate(payload)

    assert result.previous_production_metrics is None
    assert result.decision.beats_previous is True


def test_production_promotion_metadata_requires_promotion_id() -> None:
    """Ensure ProductionPromotionMetadata rejects payloads missing promotion_id."""

    payload = {
        **_promotion_base_payload(),
        "run_identity": {
            "stage": "production",
            "experiment_id": "exp-current",
            "train_run_id": "train-current",
            "eval_run_id": "eval-current",
            "explain_run_id": "explain-current",
        },
        "decision": {
            "promoted": True,
            "reason": "beats previous",
            "beats_previous": True,
        },
    }

    with pytest.raises(ValidationError, match="promotion_id"):
        ProductionPromotionMetadata.model_validate(payload)


def test_staging_promotion_metadata_requires_staging_id() -> None:
    """Ensure StagingPromotionMetadata rejects payloads missing staging_id."""

    payload = {
        **_promotion_base_payload(),
        "run_identity": {
            "stage": "staging",
            "experiment_id": "exp-current",
            "train_run_id": "train-current",
            "eval_run_id": "eval-current",
            "explain_run_id": "explain-current",
        },
        "decision": {
            "promoted": True,
            "reason": "ready for staging",
        },
    }

    with pytest.raises(ValidationError, match="staging_id"):
        StagingPromotionMetadata.model_validate(payload)


def test_staging_promotion_metadata_accepts_valid_payload() -> None:
    """Ensure StagingPromotionMetadata accepts a complete valid payload."""

    payload = {
        **_promotion_base_payload(),
        "run_identity": {
            "stage": "staging",
            "experiment_id": "exp-current",
            "train_run_id": "train-current",
            "eval_run_id": "eval-current",
            "explain_run_id": "explain-current",
            "staging_id": "staging-current",
        },
        "decision": {
            "promoted": True,
            "reason": "ready for staging",
        },
    }

    result = StagingPromotionMetadata.model_validate(payload)

    assert result.run_identity.staging_id == "staging-current"
    assert result.decision.promoted is True
