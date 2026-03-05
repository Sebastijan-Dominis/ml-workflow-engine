"""Edge-case unit tests for promotion comparison helpers."""

import pytest
from ml.exceptions import ConfigError, UserError
from ml.promotion.comparisons.production import compare_against_production_model
from ml.promotion.comparisons.thresholds import compare_against_thresholds
from ml.promotion.config.models import (
    Direction,
    MetricName,
    MetricSet,
    PromotionThresholds,
    ThresholdsConfig,
)

pytestmark = pytest.mark.unit


def _thresholds(*, direction: str = "maximize") -> PromotionThresholds:
    """Build minimal threshold configuration used by threshold-comparison tests."""
    return PromotionThresholds.model_validate(
        {
            "promotion_metrics": {
                "sets": ["val"],
                "metrics": ["f1"],
                "directions": {"f1": direction},
            },
            "thresholds": {
                "val": {"f1": 0.70},
            },
            "lineage": {
                "created_by": "tests",
                "created_at": "2026-03-05T00:00:00",
            },
        }
    )


def test_compare_against_thresholds_returns_success_when_all_thresholds_met() -> None:
    """Return success result when every configured metric satisfies threshold criteria."""
    result = compare_against_thresholds(
        evaluation_metrics={"val": {"f1": 0.72}},
        promotion_thresholds=_thresholds(),
    )

    assert result.meets_thresholds is True
    assert result.message == "All promotion criteria regarding thresholds met."


def test_compare_against_thresholds_raises_when_threshold_set_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Raise ConfigError when threshold container does not define required metric set."""
    thresholds = _thresholds()
    monkeypatch.setattr(ThresholdsConfig, "model_dump", lambda self: {})

    with pytest.raises(ConfigError, match="Thresholds for metric set 'val' are not defined"):
        compare_against_thresholds(
            evaluation_metrics={"val": {"f1": 0.72}},
            promotion_thresholds=thresholds,
        )


def test_compare_against_thresholds_raises_when_threshold_metric_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Raise ConfigError when threshold set exists but target metric threshold is missing."""
    thresholds = _thresholds()
    monkeypatch.setattr(ThresholdsConfig, "model_dump", lambda self: {"val": {}})

    with pytest.raises(ConfigError, match="Threshold value for metric 'f1' in set 'val' is not defined"):
        compare_against_thresholds(
            evaluation_metrics={"val": {"f1": 0.72}},
            promotion_thresholds=thresholds,
        )


def test_compare_against_thresholds_raises_when_direction_mapping_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Raise ConfigError when no direction is available for configured threshold metric."""
    thresholds = _thresholds()
    monkeypatch.setattr(thresholds.promotion_metrics, "directions", {})

    with pytest.raises(ConfigError, match="Direction for metric 'f1' is not defined"):
        compare_against_thresholds(
            evaluation_metrics={"val": {"f1": 0.72}},
            promotion_thresholds=thresholds,
        )


def test_compare_against_production_model_raises_when_production_metric_missing() -> None:
    """Raise UserError when production payload lacks target metric required for comparison."""
    with pytest.raises(UserError, match="Production model is missing metric 'f1' in set 'val'"):
        compare_against_production_model(
            evaluation_metrics={"val": {"f1": 0.80}},
            current_prod_model_info={"metrics": {"val": {}}},
            metric_sets=[MetricSet.VAL],
            metric_names=[MetricName.F1],
            directions={MetricName.F1: Direction.MAXIMIZE},
        )


def test_compare_against_production_model_raises_when_evaluation_metric_missing() -> None:
    """Raise UserError when evaluation payload lacks metric required for production comparison."""
    with pytest.raises(UserError, match="Evaluation metrics are missing metric 'f1' in set 'val'"):
        compare_against_production_model(
            evaluation_metrics={"val": {}},
            current_prod_model_info={"metrics": {"val": {"f1": 0.79}}},
            metric_sets=[MetricSet.VAL],
            metric_names=[MetricName.F1],
            directions={MetricName.F1: Direction.MAXIMIZE},
        )
