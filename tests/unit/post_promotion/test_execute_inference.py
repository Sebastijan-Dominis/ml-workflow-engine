import argparse
import importlib
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


def test_execute_inference_monkeypatched(monkeypatch, tmp_path: Path):
    exec_mod = importlib.import_module("ml.post_promotion.inference.execution.execute_inference")

    args = argparse.Namespace(problem="p", segment="s", snapshot_bindings_id=None)

    # Minimal features for prediction flow
    features = pd.DataFrame({"entity_id": [1, 2], "f1": [10, 20]})
    prepare_return = SimpleNamespace(features=features, entity_key="entity_id", feature_lineage=[])
    monkeypatch.setattr(exec_mod, "prepare_features", lambda *a, **kw: prepare_return)

    monkeypatch.setattr(exec_mod, "load_and_validate_artifact", lambda model_metadata: SimpleNamespace(artifact=object(), artifact_hash="ah", artifact_type="model"))

    monkeypatch.setattr(exec_mod, "predict", lambda features_for_prediction, artifact: (pd.Series([1, 0]), pd.DataFrame({"p": [0.8, 0.2]})))

    # Ensure input hashing is deterministic and cheap
    monkeypatch.setattr(exec_mod, "hash_input_row", lambda row: "h")

    # Avoid filesystem side-effects by stubbing storage
    monkeypatch.setattr(exec_mod, "store_predictions", lambda **kwargs: SimpleNamespace(cols=["run_id", "prediction"]))

    metadata_raw = {
        "problem_type": "classification",
        "segment": "s",
        "model_version": "mv",
        "model_stage": "production",
        "run_id": "rid",
        "timestamp": datetime.now().isoformat(),
        "columns": ["a", "b"],
        "snapshot_bindings_id": "sb",
        "feature_lineage": [
            {
                "name": "f",
                "version": "1",
                "snapshot_id": "s",
                "file_hash": "fh",
                "in_memory_hash": "imh",
                "feature_schema_hash": "fsh",
                "operator_hash": "oh",
                "feature_type": "tabular",
                "file_name": "fn",
                "data_format": "csv",
            }
        ],
        "artifact_type": "model",
        "artifact_hash": "ah",
        "inference_latency_seconds": 0.01,
    }

    monkeypatch.setattr(exec_mod, "prepare_metadata", lambda **kwargs: metadata_raw)

    # Let validation return a pydantic model instance
    import ml.metadata.schemas.post_promotion.infer as infer_schema

    monkeypatch.setattr(exec_mod, "validate_inference_metadata", lambda d: infer_schema.InferenceMetadata.model_validate(d))

    saved = {}

    def fake_save_metadata(metadata, target_dir):
        saved["metadata"] = metadata
        saved["target_dir"] = target_dir

    monkeypatch.setattr(exec_mod, "save_metadata", fake_save_metadata)

    # Create a minimal RegistryEntry for the call
    from ml.promotion.config.registry_entry import (
        RegistryArtifacts,
        RegistryEntry,
        RegistryEntryMetrics,
    )

    reg = RegistryEntry(
        experiment_id="e",
        train_run_id="t",
        eval_run_id="x",
        explain_run_id="r",
        model_version="mv",
        artifacts=RegistryArtifacts(model_hash="mh", model_path="mp"),
        feature_lineage=[],
        metrics=RegistryEntryMetrics(train={}, val={}, test={}),
        git_commit="c",
    )

    exec_mod.execute_inference(args=args, model_metadata=reg, stage="production", timestamp=datetime.now(), path=tmp_path, run_id="rid")

    assert saved["target_dir"] == tmp_path
    assert "artifact_hash" in saved["metadata"]
