from typing import cast

import pytest
from ml.exceptions import PipelineContractError
from ml.post_promotion.monitoring.extraction.expected_performance import get_expected_performance
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


def make_promotion_config(metric):
    return PromotionMetricsConfig(
        sets=[MetricSet.TEST],
        metrics=[metric],
        directions={metric: Direction.MAXIMIZE},
    )


def test_get_expected_performance_success():
    cfg = make_promotion_config(MetricName.ACCURACY)
    artifacts = RegistryArtifacts(model_hash="m", model_path="/tmp/m", pipeline_hash=None, pipeline_path=None)
    fl = RegistryFeatureSetLineage(
        name="f",
        version="v1",
        snapshot_id="s1",
        file_hash="fh",
        in_memory_hash="ih",
        feature_schema_hash="fs",
        operator_hash="oh",
        feature_type="tabular",
    )

    metrics = RegistryEntryMetrics(train={}, val={}, test={MetricName.ACCURACY: 0.9})

    model_metadata = RegistryEntry(
        experiment_id="e",
        train_run_id="t",
        eval_run_id="ev",
        explain_run_id="ex",
        model_version="v1",
        artifacts=artifacts,
        feature_lineage=[fl],
        metrics=metrics,
        git_commit="c",
    )

    out = get_expected_performance(model_metadata, cfg)
    assert MetricName.ACCURACY in out
    assert out[MetricName.ACCURACY] == 0.9


def test_get_expected_performance_missing_or_invalid_raises():
    cfg = make_promotion_config(MetricName.ACCURACY)
    artifacts = RegistryArtifacts(model_hash="m", model_path="/tmp/m", pipeline_hash=None, pipeline_path=None)
    fl = RegistryFeatureSetLineage(
        name="f",
        version="v1",
        snapshot_id="s1",
        file_hash="fh",
        in_memory_hash="ih",
        feature_schema_hash="fs",
        operator_hash="oh",
        feature_type="tabular",
    )

    # construct with a valid numeric value, then mutate to None to avoid type-checker errors
    metrics = RegistryEntryMetrics(train={}, val={}, test={MetricName.ACCURACY.value: 0})
    # mutate via a cast to avoid static type errors while creating a runtime-invalid value
    cast(dict[str, object], metrics.test)[MetricName.ACCURACY.value] = None

    model_metadata = RegistryEntry(
        experiment_id="e",
        train_run_id="t",
        eval_run_id="ev",
        explain_run_id="ex",
        model_version="v1",
        artifacts=artifacts,
        feature_lineage=[fl],
        metrics=metrics,
        git_commit="c",
    )

    with pytest.raises(PipelineContractError):
        get_expected_performance(model_metadata, cfg)
