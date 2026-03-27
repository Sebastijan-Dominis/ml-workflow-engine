from pathlib import Path

import pytest
from ml.exceptions import PipelineContractError
from ml.post_promotion.inference.loading.artifact import load_and_validate_artifact
from ml.promotion.config.registry_entry import (
    RegistryArtifacts,
    RegistryEntry,
    RegistryEntryMetrics,
    RegistryFeatureSetLineage,
)


def make_registry_entry(tmp_path: Path, pipeline_path: str | None, model_path: str | None, expected_hash: str):
    artifacts_obj = RegistryArtifacts(
        model_hash="mhash",
        model_path=str(model_path) if model_path is not None else str(tmp_path / "missing_model"),
        pipeline_hash=expected_hash,
        pipeline_path=str(pipeline_path) if pipeline_path is not None else str(tmp_path / "missing_pipeline"),
    )

    feature_lineage_objs = [
        RegistryFeatureSetLineage(
            name="fset",
            version="v1",
            snapshot_id="s1",
            file_hash="fh",
            in_memory_hash="ih",
            feature_schema_hash="fs",
            operator_hash="oh",
            feature_type="tabular",
        )
    ]

    metrics_obj = RegistryEntryMetrics(train={}, val={}, test={})

    return RegistryEntry(
        experiment_id="e",
        train_run_id="tr",
        eval_run_id="er",
        explain_run_id="xr",
        model_version="v1",
        artifacts=artifacts_obj,
        feature_lineage=feature_lineage_objs,
        metrics=metrics_obj,
        git_commit="c1",
    )


def test_load_and_validate_artifact_pipeline_success(tmp_path, monkeypatch):
    pipeline_file = tmp_path / "pipeline.bin"
    pipeline_file.write_text("ok")
    expected_hash = "abc123"

    reg = make_registry_entry(tmp_path, pipeline_path=str(pipeline_file), model_path=None, expected_hash=expected_hash)

    import ml.post_promotion.inference.loading.artifact as artifact_mod

    monkeypatch.setattr(artifact_mod, "load_model_or_pipeline", lambda p, target_type: "pipeline_obj")
    monkeypatch.setattr(artifact_mod, "hash_artifact", lambda p: expected_hash)

    ret = load_and_validate_artifact(reg)
    assert ret.artifact == "pipeline_obj"
    assert ret.artifact_hash == expected_hash
    assert ret.artifact_type == "pipeline"


def test_load_and_validate_artifact_model_success(tmp_path, monkeypatch):
    model_file = tmp_path / "model.bin"
    model_file.write_text("ok")
    expected_hash = "modelhash"

    # point pipeline_path to a non-existing path so model path branch runs
    reg = make_registry_entry(tmp_path, pipeline_path=str(tmp_path / "no_pipeline"), model_path=str(model_file), expected_hash=expected_hash)

    import ml.post_promotion.inference.loading.artifact as artifact_mod

    monkeypatch.setattr(artifact_mod, "load_model_or_pipeline", lambda p, target_type: "model_obj")
    monkeypatch.setattr(artifact_mod, "hash_artifact", lambda p: expected_hash)

    ret = load_and_validate_artifact(reg)
    assert ret.artifact == "model_obj"
    assert ret.artifact_hash == expected_hash
    assert ret.artifact_type == "model"


def test_load_and_validate_artifact_no_artifact_raises(tmp_path):
    reg = make_registry_entry(tmp_path, pipeline_path=str(tmp_path / "no_pipeline"), model_path=str(tmp_path / "no_model"), expected_hash="x")
    with pytest.raises(PipelineContractError):
        load_and_validate_artifact(reg)


def test_load_and_validate_artifact_hash_mismatch(tmp_path, monkeypatch):
    model_file = tmp_path / "model.bin"
    model_file.write_text("ok")
    expected_hash = "expected"

    reg = make_registry_entry(tmp_path, pipeline_path=str(tmp_path / "no_pipeline"), model_path=str(model_file), expected_hash=expected_hash)

    import ml.post_promotion.inference.loading.artifact as artifact_mod

    monkeypatch.setattr(artifact_mod, "load_model_or_pipeline", lambda p, target_type: "model_obj")
    monkeypatch.setattr(artifact_mod, "hash_artifact", lambda p: "different")

    with pytest.raises(PipelineContractError):
        load_and_validate_artifact(reg)
