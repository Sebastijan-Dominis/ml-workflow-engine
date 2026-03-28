import argparse
from typing import cast

import ml.post_promotion.monitoring.performance.assessment as assess_mod
import pandas as pd
import pytest
from ml.exceptions import ConfigError, PipelineContractError
from ml.promotion.config.promotion_thresholds import (
    Direction,
    MetricName,
    MetricSet,
    PromotionMetricsConfig,
)
from ml.promotion.config.registry_entry import (
    RegistryArtifacts,
    RegistryEntry,
    RegistryEntryMetrics,
    RegistryFeatureSetLineage,
)


def _build_registry_entry() -> RegistryEntry:
    artifacts = RegistryArtifacts(model_hash="h", model_path="p")
    feature_lineage = [
        RegistryFeatureSetLineage(
            name="fl",
            version="v1",
            snapshot_id="s1",
            file_hash="fh",
            in_memory_hash="imh",
            feature_schema_hash="fsh",
            operator_hash="oh",
            feature_type="tabular",
        )
    ]
    metrics = RegistryEntryMetrics(train={}, val={}, test={})
    return RegistryEntry(
        experiment_id="e",
        train_run_id="tr",
        eval_run_id="er",
        explain_run_id="xr",
        model_version="v",
        artifacts=artifacts,
        feature_lineage=feature_lineage,
        metrics=metrics,
        git_commit="gc",
    )


def test_assess_maximize_degradation(monkeypatch):
    args = argparse.Namespace()
    model_metadata = _build_registry_entry()
    stage = "production"
    target = pd.Series([1, 1])

    promotion_metrics_info = PromotionMetricsConfig(
        sets=[MetricSet.TEST],
        metrics=[MetricName.ACCURACY],
        directions={MetricName.ACCURACY: Direction.MAXIMIZE},
    )

    monkeypatch.setattr(assess_mod, "get_expected_performance", lambda mm, pm: {MetricName.ACCURACY: 0.9})
    monkeypatch.setattr(assess_mod, "calculate_current_performance", lambda **kw: {MetricName.ACCURACY: 0.8})

    res = assess_mod.assess_model_performance(args=args, model_metadata=model_metadata, stage=stage, target=target, promotion_metrics_info=promotion_metrics_info)
    assert res[MetricName.ACCURACY]["status"] == "degradation"


def test_assess_maximize_acceptable(monkeypatch):
    args = argparse.Namespace()
    model_metadata = _build_registry_entry()
    stage = "staging"
    target = pd.Series([0, 1])

    promotion_metrics_info = PromotionMetricsConfig(
        sets=[MetricSet.TEST],
        metrics=[MetricName.ACCURACY],
        directions={MetricName.ACCURACY: Direction.MAXIMIZE},
    )

    monkeypatch.setattr(assess_mod, "get_expected_performance", lambda mm, pm: {MetricName.ACCURACY: 0.5})
    monkeypatch.setattr(assess_mod, "calculate_current_performance", lambda **kw: {MetricName.ACCURACY: 0.6})

    res = assess_mod.assess_model_performance(args=args, model_metadata=model_metadata, stage=stage, target=target, promotion_metrics_info=promotion_metrics_info)
    assert res[MetricName.ACCURACY]["status"] == "acceptable"


def test_assess_minimize_degradation(monkeypatch):
    args = argparse.Namespace()
    model_metadata = _build_registry_entry()
    stage = "production"
    target = pd.Series([1, 1])

    promotion_metrics_info = PromotionMetricsConfig(
        sets=[MetricSet.TEST],
        metrics=[MetricName.MAE],
        directions={MetricName.MAE: Direction.MINIMIZE},
    )

    monkeypatch.setattr(assess_mod, "get_expected_performance", lambda mm, pm: {MetricName.MAE: 0.2})
    monkeypatch.setattr(assess_mod, "calculate_current_performance", lambda **kw: {MetricName.MAE: 0.25})

    res = assess_mod.assess_model_performance(args=args, model_metadata=model_metadata, stage=stage, target=target, promotion_metrics_info=promotion_metrics_info)
    assert res[MetricName.MAE]["status"] == "degradation"


def test_assess_invalid_direction_raises(monkeypatch):
    args = argparse.Namespace()
    model_metadata = _build_registry_entry()
    stage = "production"
    target = pd.Series([1, 1])

    promotion_metrics_info = PromotionMetricsConfig(
        sets=[MetricSet.TEST],
        metrics=[MetricName.ACCURACY],
        directions={MetricName.ACCURACY: Direction.MAXIMIZE},
    )

    monkeypatch.setattr(assess_mod, "get_expected_performance", lambda mm, pm: {MetricName.ACCURACY: 0.5})
    monkeypatch.setattr(assess_mod, "calculate_current_performance", lambda **kw: {MetricName.ACCURACY: 0.6})

    # Mutate directions at runtime to simulate an invalid runtime value while keeping static typing happy
    cast(dict, promotion_metrics_info.directions)[MetricName.ACCURACY] = "invalid"

    with pytest.raises(ConfigError):
        assess_mod.assess_model_performance(args=args, model_metadata=model_metadata, stage=stage, target=target, promotion_metrics_info=promotion_metrics_info)


def test_assess_missing_current_value_raises(monkeypatch):
    args = argparse.Namespace()
    model_metadata = _build_registry_entry()
    stage = "production"
    target = pd.Series([1, 1])

    promotion_metrics_info = PromotionMetricsConfig(
        sets=[MetricSet.TEST],
        metrics=[MetricName.ACCURACY],
        directions={MetricName.ACCURACY: Direction.MAXIMIZE},
    )

    monkeypatch.setattr(assess_mod, "get_expected_performance", lambda mm, pm: {MetricName.ACCURACY: 0.5})
    monkeypatch.setattr(assess_mod, "calculate_current_performance", lambda **kw: {MetricName.ACCURACY: None})

    with pytest.raises(PipelineContractError):
        assess_mod.assess_model_performance(args=args, model_metadata=model_metadata, stage=stage, target=target, promotion_metrics_info=promotion_metrics_info)
