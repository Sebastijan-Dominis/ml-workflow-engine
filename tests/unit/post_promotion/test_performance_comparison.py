
import ml.post_promotion.monitoring.performance.comparison as comp_mod
import pytest
from ml.exceptions import ConfigError, PipelineContractError
from ml.post_promotion.monitoring.classes.function_returns import MonitoringExecutionOutput


def _make_monitoring_output(current, direction):
    return MonitoringExecutionOutput(
        drift_results={},
        performance_results={"m": {"current": current, "direction": direction}},
        model_version="v",
    )


def test_compare_improvement_maximize():
    prod = _make_monitoring_output(0.5, "maximize")
    stage = _make_monitoring_output(0.7, "maximize")

    res = comp_mod.compare_production_and_staging_performance(prod, stage)
    assert res["m"]["status"] == "improvement"


def test_compare_degradation_minimize():
    prod = _make_monitoring_output(0.3, "minimize")
    stage = _make_monitoring_output(0.1, "minimize")

    res = comp_mod.compare_production_and_staging_performance(prod, stage)
    assert res["m"]["status"] == "improvement"


def test_compare_missing_metric_skips():
    prod = MonitoringExecutionOutput(drift_results={}, performance_results={"m": {"current": 0.1, "direction": "maximize"}}, model_version="v")
    stage = MonitoringExecutionOutput(drift_results={}, performance_results={}, model_version="v")

    res = comp_mod.compare_production_and_staging_performance(prod, stage)
    assert res == {}


def test_compare_non_numeric_raises():
    prod = _make_monitoring_output("bad", "maximize")
    stage = _make_monitoring_output(0.1, "maximize")

    with pytest.raises(PipelineContractError):
        comp_mod.compare_production_and_staging_performance(prod, stage)


def test_compare_direction_mismatch_raises():
    prod = _make_monitoring_output(0.5, "maximize")
    stage = _make_monitoring_output(0.6, "minimize")

    with pytest.raises(ConfigError):
        comp_mod.compare_production_and_staging_performance(prod, stage)
