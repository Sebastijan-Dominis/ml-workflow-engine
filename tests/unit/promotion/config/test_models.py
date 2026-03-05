"""Unit tests for the PromotionThresholds schema in ml.promotion.config.models. The tests verify that the schema correctly accepts consistent payloads, raises ConfigError for mismatched metric sets and mismatched metrics in sets, and properly parses the promotion metrics and thresholds. A helper function is included to generate a valid base payload for testing the PromotionThresholds schema."""
import pytest
from ml.exceptions import ConfigError
from ml.promotion.config.models import PromotionThresholds

pytestmark = pytest.mark.unit


def _base_payload() -> dict:
    """Helper function to create a valid base payload for testing the PromotionThresholds schema.

    Returns:
        dict: A valid payload dictionary for testing the PromotionThresholds schema.
    """
    return {
        "promotion_metrics": {
            "sets": ["val", "test"],
            "metrics": ["f1", "roc_auc"],
            "directions": {
                "f1": "maximize",
                "roc_auc": "maximize",
            },
        },
        "thresholds": {
            "val": {
                "f1": 0.70,
                "roc_auc": 0.80,
            },
            "test": {
                "f1": 0.69,
                "roc_auc": 0.79,
            },
        },
        "lineage": {
            "created_by": "tests",
            "created_at": "2026-03-05T00:00:00",
        },
    }


def test_promotion_thresholds_model_accepts_consistent_payload() -> None:
    """Test that the PromotionThresholds schema accepts a consistent payload and correctly parses the promotion metrics and thresholds."""
    config = PromotionThresholds.model_validate(_base_payload())

    assert [metric.value for metric in config.promotion_metrics.metrics] == ["f1", "roc_auc"]
    assert config.thresholds.val["f1"] == pytest.approx(0.70)
    assert config.thresholds.test["roc_auc"] == pytest.approx(0.79)


def test_promotion_thresholds_model_rejects_mismatched_metric_sets() -> None:
    """Test that the PromotionThresholds schema raises a ConfigError if the sets defined in the promotion metrics do not match the sets defined in the thresholds."""
    payload = _base_payload()
    payload["promotion_metrics"]["sets"] = ["val"]

    with pytest.raises(ConfigError, match="do not match threshold sets"):
        PromotionThresholds.model_validate(payload)


def test_promotion_thresholds_model_rejects_mismatched_metrics_in_set() -> None:
    """Test that the PromotionThresholds schema raises a ConfigError if the metrics defined in the promotion metrics do not match the metrics defined in the thresholds for a given set."""
    payload = _base_payload()
    payload["thresholds"]["test"] = {"f1": 0.69}

    with pytest.raises(ConfigError, match="do not match threshold metrics"):
        PromotionThresholds.model_validate(payload)
