"""Unit tests for the compare_against_thresholds function in ml.promotion.comparisons.thresholds. The tests verify that the function correctly identifies when evaluation metrics meet or do not meet specified promotion thresholds, and that it raises appropriate errors when required metrics are missing from the evaluation metrics. A helper function is included to create PromotionThresholds instances with specified threshold values for testing purposes."""
import pytest
from ml.exceptions import UserError
from ml.promotion.comparisons.thresholds import compare_against_thresholds
from ml.promotion.config.models import PromotionThresholds

pytestmark = pytest.mark.unit


def _build_thresholds(*, threshold_value: float = 0.70) -> PromotionThresholds:
    """Helper function to build a PromotionThresholds instance with a specified threshold value for testing.

    Args:
        threshold_value (float): The threshold value to set for the 'f1' metric in the 'val' set. Defaults to 0.70.

    Returns:
        PromotionThresholds: An instance of PromotionThresholds with the specified threshold value.
    """
    return PromotionThresholds.model_validate(
        {
            "promotion_metrics": {
                "sets": ["val"],
                "metrics": ["f1"],
                "directions": {"f1": "maximize"},
            },
            "thresholds": {
                "val": {"f1": threshold_value},
            },
            "lineage": {
                "created_by": "tests",
                "created_at": "2026-03-05T00:00:00",
            },
        }
    )


def test_compare_against_thresholds_returns_failure_result_when_threshold_not_met() -> None:
    """Test that compare_against_thresholds returns a result indicating failure when the evaluation metric does not meet the specified promotion threshold."""
    thresholds = _build_thresholds(threshold_value=0.80)

    result = compare_against_thresholds(
        evaluation_metrics={"val": {"f1": 0.80}},
        promotion_thresholds=thresholds,
    )

    assert result.meets_thresholds is False
    assert "does not meet the promotion threshold" in result.message
    assert list(result.target_sets) == ["val"]
    assert list(result.target_metrics) == ["f1"]


def test_compare_against_thresholds_raises_user_error_when_metric_is_missing() -> None:
    """Test that compare_against_thresholds raises a UserError when the evaluation metrics do not include a metric that is specified in the promotion thresholds."""
    thresholds = _build_thresholds(threshold_value=0.70)

    with pytest.raises(UserError, match="Evaluation metric 'f1' is not available"):
        compare_against_thresholds(
            evaluation_metrics={"val": {}},
            promotion_thresholds=thresholds,
        )
