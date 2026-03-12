"""Unit tests for search and promotion metadata schema models.

These tests focus on nested-structure validation and stage-specific payload
requirements in SearchRecord and promotion metadata schemas.
"""

import importlib
import sys
import types

import pytest
from pydantic import ValidationError

# Search metadata imports ml.types package, which imports catboost at import time.
if "catboost" not in sys.modules:
    catboost_stub = types.ModuleType("catboost")
    catboost_stub.__dict__.update(
        {
            "CatBoostClassifier": type("CatBoostClassifier", (), {}),
            "CatBoostRegressor": type("CatBoostRegressor", (), {}),
        }
    )
    sys.modules["catboost"] = catboost_stub

SearchRecord = importlib.import_module("ml.metadata.schemas.search.search").SearchRecord
ProductionPromotionMetadata = importlib.import_module(
    "ml.metadata.schemas.promotion.promote"
).ProductionPromotionMetadata
StagingPromotionMetadata = importlib.import_module(
    "ml.metadata.schemas.promotion.promote"
).StagingPromotionMetadata


pytestmark = pytest.mark.unit


def _search_metadata_payload() -> dict:
    """Return a valid SearchRecord.metadata payload."""

    return {
        "problem": "cancellation",
        "segment": "city_hotel_online_ta",
        "version": "v1",
        "experiment_id": "exp-001",
        "sources": {"main": "defaults", "extends": ["segment"]},
        "env": "dev",
        "best_params_path": "experiments/cancellation/best_params.json",
        "algorithm": "catboost",
        "pipeline_version": "v1",
        "created_by": "tests",
        "created_at": "2026-03-05T12:00:00",
        "owner": "ml-team",
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
        "seed": 42,
        "hardware": {"task_type": "CPU", "devices": []},
        "git_commit": "abc123",
        "config_hash": "cfg-hash",
        "validation_status": "ok",
        "pipeline_hash": "pipe-hash",
        "scoring_method": "roc_auc",
        "splits_info": {
            "train": {"n_rows": 1000, "class_distribution": {"0": 700, "1": 300}, "positive_rate": 0.3},
            "val": {"n_rows": 200, "class_distribution": {"0": 140, "1": 60}, "positive_rate": 0.3},
            "test": {"n_rows": 200, "class_distribution": {"0": 140, "1": 60}, "positive_rate": 0.3},
        },
        "target_transform": None,
        "class_weighting": None,
    }


def _promotion_base_payload() -> dict:
    """Return a valid shared promotion metadata payload."""

    return {
        "previous_production_run_identity": {
            "experiment_id": "exp-prev",
            "train_run_id": "train-prev",
            "eval_run_id": "eval-prev",
            "explain_run_id": "explain-prev",
            "promotion_id": "promotion-prev",
        },
        "metrics": {
            "train": {"f1": 0.8, "roc_auc": 0.85},
            "val": {"f1": 0.78, "roc_auc": 0.83},
            "test": {"f1": 0.77, "roc_auc": 0.82},
        },
        "previous_production_metrics": {
            "train": {"f1": 0.75, "roc_auc": 0.8},
            "val": {"f1": 0.74, "roc_auc": 0.79},
            "test": {"f1": 0.73, "roc_auc": 0.78},
        },
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


def test_search_record_accepts_valid_payload() -> None:
    """Ensure SearchRecord accepts a complete valid nested payload."""

    payload = {
        "metadata": _search_metadata_payload(),
        "config": {"random_state": 42},
        "search_results": {
            "best_pipeline_params": {"depth": 6},
            "best_model_params": {"learning_rate": 0.05},
            "phases": {"broad": {"n_iter": 10}, "narrow": {"n_iter": 5}},
        },
    }

    result = SearchRecord.model_validate(payload)

    assert result.metadata.experiment_id == "exp-001"
    assert result.search_results.best_model_params["learning_rate"] == 0.05


def test_search_record_rejects_unknown_phase_fields() -> None:
    """Ensure SearchRecord rejects extra keys in the phases payload."""

    payload = {
        "metadata": _search_metadata_payload(),
        "config": {},
        "search_results": {
            "best_pipeline_params": {},
            "best_model_params": {},
            "phases": {"broad": {}, "unexpected": {}},
        },
    }

    with pytest.raises(ValidationError):
        SearchRecord.model_validate(payload)


def test_production_promotion_metadata_accepts_valid_payload() -> None:
    """Ensure ProductionPromotionMetadata accepts valid production-stage payload."""

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

    assert result.run_identity.stage == "production"
    assert result.decision.beats_previous is True


def test_staging_promotion_metadata_accepts_valid_payload() -> None:
    """Ensure StagingPromotionMetadata accepts valid staging-stage payload."""

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

    assert result.run_identity.stage == "staging"
    assert result.decision.promoted is True


def test_production_promotion_metadata_rejects_missing_beats_previous() -> None:
    """Ensure production promotion metadata requires `beats_previous` decision field."""

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
        },
    }

    with pytest.raises(ValidationError, match="beats_previous"):
        ProductionPromotionMetadata.model_validate(payload)
