"""Unit tests for promotion comparison against current production model."""

import pytest
from ml.exceptions import ConfigError, UserError
from ml.promotion.comparisons.production import compare_against_production_model
from ml.promotion.config.models import Direction, MetricName, MetricSet

pytestmark = pytest.mark.unit


def _comparison_inputs() -> tuple[list[MetricSet], list[MetricName], dict[MetricName, Direction]]:
    """Build a minimal shared metric-set/metric/direction configuration."""
    metric_sets = [MetricSet.VAL]
    metric_names = [MetricName.F1]
    directions = {MetricName.F1: Direction.MAXIMIZE}
    return metric_sets, metric_names, directions


def test_compare_against_production_model_skips_when_no_production_model() -> None:
    """Return a positive skip result when no production model is currently registered."""
    metric_sets, metric_names, directions = _comparison_inputs()

    result = compare_against_production_model(
        evaluation_metrics={"val": {"f1": 0.80}},
        current_prod_model_info=None,
        metric_sets=metric_sets,
        metric_names=metric_names,
        directions=directions,
    )

    assert result.beats_previous is True
    assert "Skipping comparison" in result.message
    assert result.previous_production_metrics is None


def test_compare_against_production_model_raises_when_registered_model_has_no_metrics() -> None:
    """Raise UserError if production registry payload does not contain metrics."""
    metric_sets, metric_names, directions = _comparison_inputs()

    with pytest.raises(UserError, match="does not have metrics information"):
        compare_against_production_model(
            evaluation_metrics={"val": {"f1": 0.80}},
            current_prod_model_info={"model_uri": "x"},
            metric_sets=metric_sets,
            metric_names=metric_names,
            directions=directions,
        )


def test_compare_against_production_model_raises_when_direction_is_missing() -> None:
    """Raise ConfigError when no optimization direction is defined for a target metric."""
    metric_sets, metric_names, _ = _comparison_inputs()

    with pytest.raises(ConfigError, match="Direction for metric 'f1' is not defined"):
        compare_against_production_model(
            evaluation_metrics={"val": {"f1": 0.81}},
            current_prod_model_info={"metrics": {"val": {"f1": 0.79}}},
            metric_sets=metric_sets,
            metric_names=metric_names,
            directions={},
        )


def test_compare_against_production_model_returns_failure_when_metric_does_not_outperform() -> None:
    """Return a non-promotion result when at least one metric fails production baseline."""
    metric_sets, metric_names, directions = _comparison_inputs()
    prod_metrics = {"val": {"f1": 0.80}}

    result = compare_against_production_model(
        evaluation_metrics={"val": {"f1": 0.80}},
        current_prod_model_info={"metrics": prod_metrics},
        metric_sets=metric_sets,
        metric_names=metric_names,
        directions=directions,
    )

    assert result.beats_previous is False
    assert "does not outperform production model" in result.message
    assert result.previous_production_metrics == prod_metrics


def test_compare_against_production_model_returns_success_when_all_metrics_outperform() -> None:
    """Return a positive result when all configured metrics beat production baselines."""
    metric_sets, metric_names, directions = _comparison_inputs()
    prod_metrics = {"val": {"f1": 0.79}}

    result = compare_against_production_model(
        evaluation_metrics={"val": {"f1": 0.81}},
        current_prod_model_info={"metrics": prod_metrics},
        metric_sets=metric_sets,
        metric_names=metric_names,
        directions=directions,
    )

    assert result.beats_previous is True
    assert result.message == "Model outperforms production model on all metrics."
    assert result.previous_production_metrics == prod_metrics
