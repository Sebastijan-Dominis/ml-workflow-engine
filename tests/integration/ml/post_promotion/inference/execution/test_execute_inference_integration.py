"""Integration test for `ml.post_promotion.inference.execution.execute_inference`.

This test stubs external dependencies (artifact loading, feature preparation,
prediction, storing and metadata validation) to exercise the orchestration
logic without heavy I/O or optional native dependencies.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
from ml.post_promotion.inference.execution.execute_inference import execute_inference


def test_execute_inference_orchestrates_flow(tmp_path: Path, monkeypatch: Any) -> None:
    # Prepare fake features return
    df = pd.DataFrame({"id": [1, 2], "f1": [0.1, 0.2]})
    prep_ret = SimpleNamespace(features=df, entity_key="id", feature_lineage=[])

    import importlib
    execute_module = importlib.import_module(
        "ml.post_promotion.inference.execution.execute_inference"
    )

    monkeypatch.setattr(execute_module, "prepare_features", lambda *args, **kwargs: prep_ret)

    # Fake artifact loading
    fake_artifact = SimpleNamespace(predict=lambda X: [0, 1], predict_proba=lambda X: [[0.1, 0.9], [0.8, 0.2]])
    monkeypatch.setattr(
        execute_module,
        "load_and_validate_artifact",
        lambda model_metadata: SimpleNamespace(artifact=fake_artifact, artifact_hash="ahash", artifact_type="model"),
    )

    # Use the real hash_input_row for deterministic hashing

    # Fake predict (module-level function) - keep to default by not monkeypatching

    # Stub store_predictions to avoid parquet/io and return cols
    monkeypatch.setattr(
        execute_module,
        "store_predictions",
        lambda *args, **kwargs: SimpleNamespace(cols=["run_id", "entity_id", "prediction"]),
    )

    # prepare_metadata -> return raw dict
    monkeypatch.setattr(execute_module, "prepare_metadata", lambda **kwargs: {"meta": "ok"})

    # validate_inference_metadata -> return object with model_dump()
    monkeypatch.setattr(
        execute_module,
        "validate_inference_metadata",
        lambda raw: SimpleNamespace(model_dump=lambda exclude_none=True: {"validated": True}),
    )

    saved = {}

    def fake_save_metadata(*, metadata, target_dir: Path):
        saved["metadata"] = metadata
        saved["target_dir"] = target_dir

    monkeypatch.setattr(execute_module, "save_metadata", fake_save_metadata)

    # Build dummy args and model metadata (lightweight); cast to expected type
    args = cast(Any, SimpleNamespace(snapshot_bindings_id="snap-1"))
    model_metadata = cast(Any, SimpleNamespace(model_version="v1"))

    # Execute
    execute_inference(
        args=args,
        model_metadata=model_metadata,
        stage="production",
        timestamp=datetime.utcnow(),
        path=tmp_path,
        run_id="r1",
    )

    # Assertions: save_metadata was called with validated metadata and correct path
    assert saved.get("metadata") == {"validated": True}
    assert saved.get("target_dir") == tmp_path
