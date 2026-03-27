import argparse
from datetime import datetime

from ml.modeling.models.feature_lineage import FeatureLineage
from ml.post_promotion.inference.persistence.prepare_metadata import prepare_metadata
from ml.promotion.config.registry_entry import (
    RegistryArtifacts,
    RegistryEntry,
    RegistryEntryMetrics,
    RegistryFeatureSetLineage,
)


def test_prepare_metadata_basic_post_promotion():
    args = argparse.Namespace(problem="classification", segment="all")
    artifacts = RegistryArtifacts(model_hash="m", model_path="/tmp/m", pipeline_hash=None, pipeline_path=None)
    fl_entry = RegistryFeatureSetLineage(
        name="f",
        version="v1",
        snapshot_id="s1",
        file_hash="fh",
        in_memory_hash="ih",
        feature_schema_hash="fs",
        operator_hash="oh",
        feature_type="tabular",
    )
    metrics = RegistryEntryMetrics(train={}, val={}, test={})

    model_metadata = RegistryEntry(
        experiment_id="e",
        train_run_id="t",
        eval_run_id="ev",
        explain_run_id="ex",
        model_version="v1",
        artifacts=artifacts,
        feature_lineage=[fl_entry],
        metrics=metrics,
        git_commit="c",
    )
    run_id = "rid"
    timestamp = datetime.utcnow()
    stage = "production"
    cols = ["run_id", "prediction"]
    snapshot_bindings_id = "sb"

    fl = FeatureLineage(
        name="f",
        version="v1",
        snapshot_id="s1",
        file_hash="fh",
        in_memory_hash="ih",
        feature_schema_hash="fs",
        operator_hash="oh",
        feature_type="tabular",
        file_name="fn",
        data_format="csv",
    )

    meta = prepare_metadata(
        model_metadata=model_metadata,
        args=args,
        run_id=run_id,
        timestamp=timestamp,
        stage=stage,
        cols=cols,
        snapshot_bindings_id=snapshot_bindings_id,
        feature_lineage=[fl],
        artifact_type="model",
        artifact_hash="ah",
        inference_latency_seconds=0.123,
    )

    assert meta["model_version"] == "v1"
    assert meta["problem_type"] == "classification"
    assert isinstance(meta["feature_lineage"], list)
    assert meta["artifact_hash"] == "ah"
