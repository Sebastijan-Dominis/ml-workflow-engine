import pytest
from ml.exceptions import ConfigError
from ml.promotion.config.models import PromotionThresholds

pytestmark = pytest.mark.unit


def _base_payload() -> dict:
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
    config = PromotionThresholds.model_validate(_base_payload())

    assert [metric.value for metric in config.promotion_metrics.metrics] == ["f1", "roc_auc"]
    assert config.thresholds.val["f1"] == pytest.approx(0.70)
    assert config.thresholds.test["roc_auc"] == pytest.approx(0.79)


def test_promotion_thresholds_model_rejects_mismatched_metric_sets() -> None:
    payload = _base_payload()
    payload["promotion_metrics"]["sets"] = ["val"]

    with pytest.raises(ConfigError, match="do not match threshold sets"):
        PromotionThresholds.model_validate(payload)


def test_promotion_thresholds_model_rejects_mismatched_metrics_in_set() -> None:
    payload = _base_payload()
    payload["thresholds"]["test"] = {"f1": 0.69}

    with pytest.raises(ConfigError, match="do not match threshold metrics"):
        PromotionThresholds.model_validate(payload)
