from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest
from ml.exceptions import PipelineContractError
from ml.post_promotion.inference.persistence.store_predictions import store_predictions
from ml.promotion.config.registry_entry import (
    RegistryArtifacts,
    RegistryEntry,
    RegistryEntryMetrics,
    RegistryFeatureSetLineage,
)


def test_store_predictions_entity_key_missing_raises(tmp_path):
    df = pd.DataFrame({"not_entity": ["a"]})

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
    metrics = RegistryEntryMetrics(train={}, val={}, test={})

    reg = RegistryEntry(
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
        store_predictions(
            features=df,
            entity_key="entity_key",
            run_id="r",
            input_hash=pd.Series(["h"]),
            path=tmp_path,
            timestamp=pd.Timestamp.now(),
            predictions=pd.Series([0]),
            probabilities=pd.DataFrame(),
            model_metadata=reg,
            stage="production",
        )


def test_store_predictions_with_empty_probabilities(tmp_path, monkeypatch):
    df = pd.DataFrame({"entity_key": ["e1", "e2"], "prediction": [0, 1]})
    out_dir = tmp_path / "out"

    def fake_write_table(table, path, **kwargs):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("ok")

    monkeypatch.setattr(pq, "write_table", fake_write_table)

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
    metrics = RegistryEntryMetrics(train={}, val={}, test={})

    reg = RegistryEntry(
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

    ret = store_predictions(
        features=df,
        entity_key="entity_key",
        run_id="r",
        input_hash=pd.Series(["h1", "h2"]),
        path=out_dir,
        timestamp=pd.Timestamp.now(),
        predictions=pd.Series([0, 1]),
        probabilities=pd.DataFrame(),
        model_metadata=reg,
        stage="staging",
    )

    assert ret.file_path.exists()
