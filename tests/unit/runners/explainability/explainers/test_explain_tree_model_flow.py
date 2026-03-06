"""Unit tests for tree-model explainability runner orchestration."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.unit


def _import_tree_model_module_with_registry_stub():
    """Import tree-model explainer module with lightweight registry stub to avoid cycles."""
    module_name = "ml.runners.explainability.explainers.tree_model.tree_model"
    registries_name = "ml.registries"
    registries_catalogs_name = "ml.registries.catalogs"

    sys.modules.pop(module_name, None)

    fake_registries = types.ModuleType(registries_name)
    fake_catalogs = types.ModuleType(registries_catalogs_name)
    fake_catalogs.__dict__["OP_MAP"] = {}
    fake_registries.__dict__["catalogs"] = fake_catalogs

    sys.modules[registries_name] = fake_registries
    sys.modules[registries_catalogs_name] = fake_catalogs

    return importlib.import_module(module_name)


def _model_cfg_stub() -> Any:
    """Build minimal model config object required by explain orchestration."""
    return SimpleNamespace(
        feature_store=SimpleNamespace(path="feature_store", feature_sets=["booking_context_features"]),
        split=SimpleNamespace(),
        data_type="tabular",
        task=SimpleNamespace(type="classification", subtype="binary"),
    )


def test_explain_raises_when_training_metadata_lacks_pipeline_path(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Fail fast with actionable error when training metadata has no pipeline path."""
    model_cfg = _model_cfg_stub()
    training_metadata = SimpleNamespace(artifacts=SimpleNamespace(pipeline_path=None))
    tree_model_module = _import_tree_model_module_with_registry_stub()

    monkeypatch.setattr(tree_model_module, "load_json", lambda path: {"raw": "metadata"})
    monkeypatch.setattr(tree_model_module, "validate_training_metadata", lambda raw: training_metadata)

    with caplog.at_level(
        "ERROR",
        logger="ml.runners.explainability.explainers.tree_model.tree_model",
    ), pytest.raises(
        ValueError,
        match="Training metadata is missing the path to the trained pipeline artifact",
    ):
        tree_model_module.ExplainTreeModel().explain(
            model_cfg=model_cfg,  # type: ignore[arg-type]
            train_dir=Path("experiments") / "train" / "run-1",
            top_k=10,
        )

    assert "Cannot proceed with explainability without the pipeline" in caplog.text


def test_explain_runs_full_orchestration_and_returns_expected_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Execute full explain flow and return dataclass output with computed metrics."""
    model_cfg = _model_cfg_stub()
    training_metadata = SimpleNamespace(artifacts=SimpleNamespace(pipeline_path="artifacts/pipeline.joblib"))
    feature_lineage = [SimpleNamespace(model_dump=lambda: {"feature": "adr"})]
    split_obj = SimpleNamespace(X_test=pd.DataFrame({"adr": [100.0, 120.0], "lead_time": [5, 7]}))
    calls: list[str] = []
    tree_model_module = _import_tree_model_module_with_registry_stub()

    class _PipelineStub:
        def __getitem__(self, idx: int) -> object:
            calls.append(f"pipeline_getitem:{idx}")
            return "final-model"

    pipeline_stub = _PipelineStub()

    def _load_json(path: Path) -> dict[str, str]:
        calls.append(f"load_json:{path.name}")
        return {"raw": "metadata"}

    def _validate_training_metadata(raw: dict[str, str]) -> Any:
        _ = raw
        calls.append("validate_training_metadata")
        return training_metadata

    def _load_model_or_pipeline(path: Path, kind: str) -> object:
        calls.append(f"load_model_or_pipeline:{path}:{kind}")
        return pipeline_stub

    def _get_snapshot_binding(metadata: Any) -> list[str]:
        _ = metadata
        calls.append("get_snapshot_binding")
        return ["snapshot-a"]

    def _resolve_feature_snapshots(**kwargs: Any) -> dict[str, str]:
        _ = kwargs
        calls.append("resolve_feature_snapshots")
        return {"booking_context_features": "snapshot-a"}

    def _load_features_and_target(cfg: Any, snapshot_selection: Any, strict: bool):
        _ = (cfg, snapshot_selection)
        calls.append(f"load_features_and_target:strict={strict}")
        return (
            pd.DataFrame({"adr": [100.0, 120.0], "lead_time": [5, 7]}),
            pd.Series([1, 0]),
            feature_lineage,
        )

    def _get_splits(**kwargs: Any):
        _ = kwargs
        calls.append("get_splits")
        return split_obj, {"train_rows": 1}

    def _validate_snapshot_ids(lineage: Any, selection: Any) -> None:
        _ = (lineage, selection)
        calls.append("validate_snapshot_ids")

    def _get_names_and_transformed(pipeline: object, x_test: pd.DataFrame):
        _ = (pipeline, x_test)
        calls.append("get_feature_names_and_transformed_features")
        return (
            np.array(["adr", "lead_time"], dtype=np.str_),
            pd.DataFrame({"adr": [100.0, 120.0], "lead_time": [5, 7]}),
        )

    def _get_tree_model_adapter(model: object) -> str:
        calls.append(f"get_tree_model_adapter:{model}")
        return "adapter"

    def _get_feature_importances(**kwargs: Any) -> pd.DataFrame:
        calls.append(f"get_feature_importances:top_k={kwargs['top_k']}")
        return pd.DataFrame({"feature": ["adr"], "importance": [0.8]})

    def _get_shap_importances(**kwargs: Any) -> pd.DataFrame:
        calls.append(f"get_shap_importances:top_k={kwargs['top_k']}")
        return pd.DataFrame({"feature": ["lead_time"], "mean_abs_shap": [0.9]})

    monkeypatch.setattr(tree_model_module, "load_json", _load_json)
    monkeypatch.setattr(tree_model_module, "validate_training_metadata", _validate_training_metadata)
    monkeypatch.setattr(tree_model_module, "load_model_or_pipeline", _load_model_or_pipeline)
    monkeypatch.setattr(
        tree_model_module,
        "get_snapshot_binding_from_training_metadata",
        _get_snapshot_binding,
    )
    monkeypatch.setattr(tree_model_module, "resolve_feature_snapshots", _resolve_feature_snapshots)
    monkeypatch.setattr(tree_model_module, "load_features_and_target", _load_features_and_target)
    monkeypatch.setattr(tree_model_module, "get_splits", _get_splits)
    monkeypatch.setattr(tree_model_module, "validate_snapshot_ids", _validate_snapshot_ids)
    monkeypatch.setattr(
        tree_model_module,
        "get_feature_names_and_transformed_features",
        _get_names_and_transformed,
    )
    monkeypatch.setattr(tree_model_module, "get_tree_model_adapter", _get_tree_model_adapter)
    monkeypatch.setattr(tree_model_module, "get_feature_importances", _get_feature_importances)
    monkeypatch.setattr(tree_model_module, "get_shap_importances", _get_shap_importances)

    result = tree_model_module.ExplainTreeModel().explain(
        model_cfg=model_cfg,  # type: ignore[arg-type]
        train_dir=Path("experiments") / "train" / "run-2",
        top_k=15,
    )

    assert result.feature_lineage is feature_lineage
    assert result.explainability_metrics.top_k_feature_importances is not None
    assert result.explainability_metrics.top_k_shap_importances is not None
    assert result.explainability_metrics.top_k_feature_importances.to_dict(orient="records") == [
        {"feature": "adr", "importance": 0.8}
    ]
    assert result.explainability_metrics.top_k_shap_importances.to_dict(orient="records") == [
        {"feature": "lead_time", "mean_abs_shap": 0.9}
    ]
    assert "load_json:metadata.json" in calls
    assert "validate_training_metadata" in calls
    assert "resolve_feature_snapshots" in calls
    assert "load_features_and_target:strict=True" in calls
    assert "get_splits" in calls
    assert "validate_snapshot_ids" in calls
    assert "get_feature_names_and_transformed_features" in calls
    assert "pipeline_getitem:-1" in calls
    assert "get_tree_model_adapter:final-model" in calls
    assert "get_feature_importances:top_k=15" in calls
    assert "get_shap_importances:top_k=15" in calls
