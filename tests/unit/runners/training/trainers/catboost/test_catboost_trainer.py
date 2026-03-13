"""Unit tests for CatBoost trainer orchestration contract."""

from __future__ import annotations

import importlib
import sys
import types
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest

pytestmark = pytest.mark.unit


def _import_trainer_module() -> types.ModuleType:
    """Import trainer module with an isolated feature-loader dependency."""
    module_name = "ml.runners.training.trainers.catboost.catboost"
    loader_module_name = "ml.features.loading.features_and_target"
    build_module_name = "ml.modeling.catboost.build_pipeline_with_model"
    model_specific_module_name = "ml.runners.training.utils.model_specific.catboost"

    sys.modules.pop(module_name, None)
    original_loader = sys.modules.get(loader_module_name)
    original_build_module = sys.modules.get(build_module_name)
    original_model_specific = sys.modules.get(model_specific_module_name)

    fake_loader = types.ModuleType(loader_module_name)
    fake_loader.__dict__["load_features_and_target"] = lambda *args, **kwargs: None
    sys.modules[loader_module_name] = fake_loader

    fake_build_module = types.ModuleType(build_module_name)
    fake_build_module.__dict__["build_pipeline_with_model"] = lambda **kwargs: None
    sys.modules[build_module_name] = fake_build_module

    fake_model_specific = types.ModuleType(model_specific_module_name)
    fake_model_specific.__dict__["prepare_model"] = lambda **kwargs: None
    sys.modules[model_specific_module_name] = fake_model_specific

    try:
        return importlib.import_module(module_name)
    finally:
        if original_loader is None:
            sys.modules.pop(loader_module_name, None)
        else:
            sys.modules[loader_module_name] = original_loader

        if original_build_module is None:
            sys.modules.pop(build_module_name, None)
        else:
            sys.modules[build_module_name] = original_build_module

        if original_model_specific is None:
            sys.modules.pop(model_specific_module_name, None)
        else:
            sys.modules[model_specific_module_name] = original_model_specific


def _build_minimal_cfg(task_type: str) -> Any:
    """Build lightweight config stub matching trainer attribute access."""
    return cast(
        Any,
        SimpleNamespace(
            task=SimpleNamespace(type=task_type),
            split=SimpleNamespace(),
            data_type="tabular",
            target=SimpleNamespace(transform=SimpleNamespace()),
            pipeline=SimpleNamespace(path="configs/pipelines/tabular/sample.yaml"),
            assumptions={
                "handles_categoricals": True,
                "supports_regression": True,
                "supports_classification": True,
            },
        ),
    )


def test_train_executes_classification_flow_and_returns_expected_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run full orchestration for classification and assert contract-level outputs."""
    module = _import_trainer_module()
    monkeypatch.setattr(module, "validate_pipeline_config_consistency", lambda actual_hash, search_dir: None)
    monkeypatch.setattr(module, "validate_pipeline_config_consistency", lambda actual_hash, search_dir: None)
    module_calls: dict[str, Any] = {}

    cfg = _build_minimal_cfg(task_type="classification")

    X = pd.DataFrame({"a": [1, 2], "cat": ["x", "y"]})
    y = pd.Series([0, 1], name="target")
    lineage = [SimpleNamespace(feature="a")]

    X_train = X.iloc[0:1].copy()
    y_train = y.iloc[0:1].copy()
    X_val = X.iloc[1:2].copy()
    y_val = y.iloc[1:2].copy()

    splits = SimpleNamespace(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)

    transformed_y_train = pd.Series([0], name="target")
    transformed_y_val = pd.Series([1], name="target")

    monkeypatch.setattr(module, "load_features_and_target", lambda model_cfg, snapshot_selection, strict: (X, y, lineage))
    monkeypatch.setattr(module, "get_splits", lambda **kwargs: (splits, {"kind": "holdout"}))

    transform_calls: list[str] = []

    def _transform_target(series: pd.Series, *, transform_config: Any, split_name: str) -> pd.Series:
        transform_calls.append(split_name)
        if split_name == "train":
            return transformed_y_train
        return transformed_y_val

    monkeypatch.setattr(module, "transform_target", _transform_target)
    monkeypatch.setattr(module, "load_schemas", lambda model_cfg, feature_lineage: ({}, []))

    cat_feature_calls: list[tuple[Any, Any, Any]] = []

    def _get_cat_features(model_cfg: Any, input_schema: Any, derived_schema: Any) -> list[str]:
        cat_feature_calls.append((model_cfg, input_schema, derived_schema))
        return ["cat"]

    monkeypatch.setattr(module, "get_cat_features", _get_cat_features)
    monkeypatch.setattr(module, "load_yaml", lambda path: {
        "name": "test_pipeline",
        "version": "v1",
        "steps": ["SchemaValidator"],
        "assumptions": {
            "handles_categoricals": True,
            "supports_regression": True,
            "supports_classification": True,
        },
        "lineage": {
            "created_by": "test_user",
            "created_at": datetime.now().isoformat(),
        },
    })
    monkeypatch.setattr(module, "compute_model_config_hash", lambda cfg_dict: "pipeline-hash-1")

    monkeypatch.setattr(
        module,
        "validate_model_feature_pipeline_contract",
        lambda model_cfg, pipeline_cfg, cat_features: module_calls.update(
            {"validate_contract": {"pipeline_cfg": pipeline_cfg, "cat_features": cat_features}}
        ),
    )

    monkeypatch.setattr(module, "compute_data_stats", lambda y_in: SimpleNamespace(minority_ratio=0.4, class_counts={0: 1, 1: 1}))
    monkeypatch.setattr(module, "resolve_class_weighting", lambda model_cfg, stats, library: {"class_weights": [1.0, 1.0]})

    monkeypatch.setattr(
        module,
        "prepare_model",
        lambda model_cfg, cat_features, class_weights, failure_management_dir: {
            "prepared": True,
            "class_weights": class_weights,
            "failure_management_dir": failure_management_dir,
        },
    )

    pipeline_obj = SimpleNamespace(steps=[("prep", object()), ("Model", object())])
    monkeypatch.setattr(module, "build_pipeline_with_model", lambda **kwargs: pipeline_obj)
    monkeypatch.setattr(module, "train_catboost_model", lambda model_cfg, steps, X_train, y_train, X_val, y_val: ("trained-model", "trained-pipeline"))
    monkeypatch.setattr(module, "compute_metrics", lambda **kwargs: {"f1": 0.75})

    trainer = module.CatBoostTrainer()
    output = trainer.train(cfg, strict=True, failure_management_dir=Path("failure_dir"), search_dir=Path("search"))

    assert output.model == "trained-model"
    assert output.pipeline == "trained-pipeline"
    assert output.metrics == {"f1": 0.75}
    assert output.lineage == lineage
    assert output.pipeline_cfg_hash == "pipeline-hash-1"
    assert transform_calls == ["train", "val"]
    assert len(cat_feature_calls) == 1
    assert module_calls["validate_contract"]["cat_features"] == ["cat"]


def test_train_skips_class_weight_resolution_for_regression(monkeypatch: pytest.MonkeyPatch) -> None:
    """Do not compute class weighting when task type is regression."""
    module = _import_trainer_module()
    monkeypatch.setattr(module, "validate_pipeline_config_consistency", lambda actual_hash, search_dir: None)
    monkeypatch.setattr(module, "validate_pipeline_config_consistency", lambda actual_hash, search_dir: None)
    cfg = _build_minimal_cfg(task_type="regression")

    X = pd.DataFrame({"a": [1.0, 2.0]})
    y = pd.Series([10.0, 20.0], name="target")
    splits = SimpleNamespace(
        X_train=X.iloc[0:1],
        y_train=y.iloc[0:1],
        X_val=X.iloc[1:2],
        y_val=y.iloc[1:2],
    )

    monkeypatch.setattr(module, "load_features_and_target", lambda model_cfg, snapshot_selection, strict: (X, y, []))
    monkeypatch.setattr(module, "get_splits", lambda **kwargs: (splits, {}))
    monkeypatch.setattr(module, "transform_target", lambda series, *, transform_config, split_name: series)
    monkeypatch.setattr(module, "load_schemas", lambda model_cfg, feature_lineage: ({}, []))
    monkeypatch.setattr(module, "get_cat_features", lambda *args, **kwargs: [])
    from datetime import datetime
    monkeypatch.setattr(module, "load_yaml", lambda path: {
        "name": "test_pipeline",
        "version": "v1",
        "steps": ["SchemaValidator"],
        "assumptions": {
            "handles_categoricals": True,
            "supports_regression": True,
            "supports_classification": True,
        },
        "lineage": {
            "created_by": "test_user",
            "created_at": datetime.now().isoformat(),
        },
    })
    monkeypatch.setattr(module, "compute_model_config_hash", lambda cfg_dict: "hash")
    monkeypatch.setattr(module, "validate_model_feature_pipeline_contract", lambda *args, **kwargs: None)

    compute_stats_calls: list[bool] = []
    resolve_weight_calls: list[bool] = []

    monkeypatch.setattr(module, "compute_data_stats", lambda y_in: compute_stats_calls.append(True))
    monkeypatch.setattr(module, "resolve_class_weighting", lambda *args, **kwargs: resolve_weight_calls.append(True))

    prepare_payload: dict[str, Any] = {}

    def _prepare_model(
        model_cfg: Any,
        *,
        cat_features: list[str],
        class_weights: dict[str, Any],
        failure_management_dir: Path,
    ) -> str:
        prepare_payload["class_weights"] = class_weights
        return "raw-model"

    monkeypatch.setattr(module, "prepare_model", _prepare_model)
    monkeypatch.setattr(module, "build_pipeline_with_model", lambda **kwargs: SimpleNamespace(steps=[("prep", object()), ("Model", object())]))
    monkeypatch.setattr(
        module,
        "train_catboost_model",
        lambda model_cfg, **kwargs: ("trained", "pipeline"),
    )
    monkeypatch.setattr(module, "compute_metrics", lambda **kwargs: {"rmse": 1.23})

    output = module.CatBoostTrainer().train(cfg, strict=False, failure_management_dir=Path("fm"), search_dir=Path("search"))

    assert output.metrics == {"rmse": 1.23}
    assert prepare_payload["class_weights"] == {}
    assert compute_stats_calls == []
    assert resolve_weight_calls == []
