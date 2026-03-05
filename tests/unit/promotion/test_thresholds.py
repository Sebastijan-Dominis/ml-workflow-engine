"""Unit tests for promotion threshold comparisons."""
import pytest
from ml.exceptions import UserError
from ml.promotion.comparisons.thresholds import compare_against_thresholds
from ml.promotion.config.models import PromotionThresholds

pytestmark = pytest.mark.unit


def _build_thresholds(*, threshold_value: float = 0.70) -> PromotionThresholds:
    """Build `PromotionThresholds` with a configurable validation F1 threshold."""
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
    """Verify failure result when evaluation metrics do not meet thresholds."""
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
    """Verify error handling for missing required evaluation metrics."""
    thresholds = _build_thresholds(threshold_value=0.70)

    with pytest.raises(UserError, match="Evaluation metric 'f1' is not available"):
        compare_against_thresholds(
            evaluation_metrics={"val": {}},
            promotion_thresholds=thresholds,
        )
