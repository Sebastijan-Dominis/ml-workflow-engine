"""Unit tests for CatBoost preparation pipeline step behavior."""

from __future__ import annotations

import importlib
import sys
import types
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

pytestmark = pytest.mark.unit


def _import_preparation_module() -> types.ModuleType:
    """Import preparation-step module with isolated heavy dependency modules."""
    module_name = "ml.search.searchers.catboost.pipeline.steps.preparation"
    cat_features_name = "ml.features.extraction.cat_features"
    features_target_name = "ml.features.loading.features_and_target"
    schemas_name = "ml.features.loading.schemas"
    splitting_name = "ml.features.splitting.splitting"
    target_transform_name = "ml.features.transforms.transform_target"
    contract_name = "ml.features.validation.validate_contract"
    cw_models_name = "ml.modeling.class_weighting.models"
    cw_resolve_name = "ml.modeling.class_weighting.resolve_class_weighting"
    metric_resolve_name = "ml.modeling.class_weighting.resolve_metric"
    stats_resolve_name = "ml.modeling.class_weighting.stats_resolver"

    sys.modules.pop(module_name, None)

    fake_cat_features = types.ModuleType(cat_features_name)
    fake_cat_features.__dict__["get_cat_features"] = lambda *_args, **_kwargs: []
    sys.modules[cat_features_name] = fake_cat_features

    fake_features_target = types.ModuleType(features_target_name)
    fake_features_target.__dict__["load_features_and_target"] = lambda *_args, **_kwargs: (None, None, [], "entity_key")
    sys.modules[features_target_name] = fake_features_target

    fake_schemas = types.ModuleType(schemas_name)
    fake_schemas.__dict__["load_schemas"] = lambda *_args, **_kwargs: (pd.DataFrame(), pd.DataFrame())
    sys.modules[schemas_name] = fake_schemas

    fake_splitting = types.ModuleType(splitting_name)
    fake_splitting.__dict__["get_splits"] = lambda *_args, **_kwargs: (None, {})
    sys.modules[splitting_name] = fake_splitting

    fake_target_transform = types.ModuleType(target_transform_name)
    fake_target_transform.__dict__["transform_target"] = lambda y, **_kwargs: y
    sys.modules[target_transform_name] = fake_target_transform

    fake_contract = types.ModuleType(contract_name)
    fake_contract.__dict__["validate_model_feature_pipeline_contract"] = lambda *_args, **_kwargs: None
    sys.modules[contract_name] = fake_contract

    fake_cw_models = types.ModuleType(cw_models_name)

    class _DataStats:
        pass

    fake_cw_models.__dict__["DataStats"] = _DataStats
    sys.modules[cw_models_name] = fake_cw_models

    fake_cw_resolve = types.ModuleType(cw_resolve_name)
    fake_cw_resolve.__dict__["resolve_class_weighting"] = lambda *_args, **_kwargs: {"class_weights": {}}
    sys.modules[cw_resolve_name] = fake_cw_resolve

    fake_metric_resolve = types.ModuleType(metric_resolve_name)
    fake_metric_resolve.__dict__["resolve_metric"] = lambda *_args, **_kwargs: "roc_auc"
    sys.modules[metric_resolve_name] = fake_metric_resolve

    fake_stats_resolve = types.ModuleType(stats_resolve_name)
    fake_stats_resolve.__dict__["compute_data_stats"] = lambda *_args, **_kwargs: object()
    sys.modules[stats_resolve_name] = fake_stats_resolve

    return importlib.import_module(module_name)


def _make_context(tmp_path: Path, *, task_type: str) -> SimpleNamespace:
    """Create a minimal preparation context stub with required fields."""
    pipeline_file = tmp_path / "pipeline.yaml"
    pipeline_file.write_text("steps: []\n", encoding="utf-8")

    return SimpleNamespace(
        model_cfg=SimpleNamespace(
            split=SimpleNamespace(strategy="random"),
            data_type="tabular",
            task=SimpleNamespace(type=task_type),
            pipeline=SimpleNamespace(path=str(pipeline_file)),
            target=SimpleNamespace(transform=SimpleNamespace(enabled=False, type=None)),
            assumptions={
                "handles_categoricals": True,
                "supports_regression": True,
                "supports_classification": True,
            },
        ),
        strict=True,
        class_weights=None,
        snapshot_binding_key=None,
    )


def test_preparation_step_populates_context_and_resolves_classification_weights(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Populate all context fields and resolve class weights for classification tasks."""
    preparation_module = _import_preparation_module()
    ctx = _make_context(tmp_path, task_type="classification")

    x_raw = pd.DataFrame({"country": ["PT", "GB"]})
    y_raw = pd.Series([1, 0])
    lineage = [SimpleNamespace(name="country")]
    splits = SimpleNamespace(X_train=pd.DataFrame({"country": ["PT"]}), y_train=pd.Series([1]))
    splits_info = {"train_rows": 1}
    input_schema = pd.DataFrame({"feature": ["country"], "dtype": ["object"]})
    derived_schema = pd.DataFrame({"feature": [], "source_operator": []})
    stats_obj = object()

    import sys as _sys
    monkeypatch.setattr(
        _sys.modules["ml.features.loading.features_and_target"],
        "load_features_and_target",
        lambda *_args, **_kwargs: (x_raw, y_raw, lineage, "entity_key"),
    )
    monkeypatch.setattr(preparation_module, "get_splits", lambda *_args, **_kwargs: (splits, splits_info))
    monkeypatch.setattr(preparation_module, "load_schemas", lambda *_args, **_kwargs: (input_schema, derived_schema))
    monkeypatch.setattr(preparation_module, "load_yaml", lambda _path: {
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
    monkeypatch.setattr(preparation_module, "compute_model_config_hash", lambda _cfg: "pipeline-hash")
    monkeypatch.setattr(preparation_module, "get_cat_features", lambda *_args, **_kwargs: ["country"])
    monkeypatch.setattr(preparation_module, "validate_model_feature_pipeline_contract", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(preparation_module, "compute_data_stats", lambda _y_train: stats_obj)
    monkeypatch.setattr(preparation_module, "resolve_metric", lambda *_args, **_kwargs: "roc_auc")
    monkeypatch.setattr(preparation_module, "resolve_class_weighting", lambda *_args, **_kwargs: {"class_weights": {0: 1.0, 1: 2.0}})
    monkeypatch.setattr(preparation_module, "transform_target", lambda y, **_kwargs: y + 10)

    result = preparation_module.PreparationStep().run(ctx)

    assert result is ctx
    assert ctx.X_train.equals(splits.X_train)
    assert ctx.y_train.tolist() == [11]
    assert ctx.splits_info == splits_info
    assert ctx.pipeline_cfg.steps == ["SchemaValidator"]
    assert ctx.input_schema.equals(input_schema)
    assert ctx.derived_schema.equals(derived_schema)
    assert ctx.cat_features == ["country"]
    assert ctx.pipeline_hash == "pipeline-hash"
    assert ctx.feature_lineage == lineage
    assert ctx.scoring == "roc_auc"
    assert ctx.class_weights == {"class_weights": {0: 1.0, 1: 2.0}}


def test_preparation_step_skips_class_weighting_for_non_classification(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Do not compute stats/class weights when task type is not classification."""
    preparation_module = _import_preparation_module()
    ctx = _make_context(tmp_path, task_type="regression")

    x_raw = pd.DataFrame({"x": [1.0, 2.0]})
    y_raw = pd.Series([10.0, 20.0])
    splits = SimpleNamespace(X_train=pd.DataFrame({"x": [1.0]}), y_train=pd.Series([10.0]))

    import sys as _sys
    monkeypatch.setattr(
        _sys.modules["ml.features.loading.features_and_target"],
        "load_features_and_target",
        lambda *_args, **_kwargs: (x_raw, y_raw, [], "entity_key"),
    )
    monkeypatch.setattr(preparation_module, "get_splits", lambda *_args, **_kwargs: (splits, {"train_rows": 1}))
    monkeypatch.setattr(preparation_module, "load_schemas", lambda *_args, **_kwargs: (pd.DataFrame(), pd.DataFrame()))
    from datetime import datetime
    monkeypatch.setattr(preparation_module, "load_yaml", lambda _path: {
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
    monkeypatch.setattr(preparation_module, "compute_model_config_hash", lambda _cfg: "h")
    monkeypatch.setattr(preparation_module, "get_cat_features", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(preparation_module, "validate_model_feature_pipeline_contract", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        preparation_module,
        "compute_data_stats",
        lambda _y_train: (_ for _ in ()).throw(AssertionError("should not be called")),
    )
    monkeypatch.setattr(preparation_module, "resolve_metric", lambda *_args, **_kwargs: "rmse")
    monkeypatch.setattr(
        preparation_module,
        "resolve_class_weighting",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not be called")),
    )
    monkeypatch.setattr(preparation_module, "transform_target", lambda y, **_kwargs: y)

    preparation_module.PreparationStep().run(ctx)

    assert ctx.scoring == "rmse"
    assert ctx.class_weights is None


def test_preparation_step_before_logs_start_message(caplog: pytest.LogCaptureFixture) -> None:
    """Emit the documented start log line from `before` hook."""
    preparation_module = _import_preparation_module()

    with caplog.at_level("INFO", logger=preparation_module.__name__):
        preparation_module.PreparationStep().before(SimpleNamespace())

    assert "Starting preparation step." in caplog.text


def test_preparation_step_after_logs_completion_message(caplog: pytest.LogCaptureFixture) -> None:
    """Emit the documented completion log line from `after` hook."""
    preparation_module = _import_preparation_module()

    with caplog.at_level("INFO", logger=preparation_module.__name__):
        preparation_module.PreparationStep().after(SimpleNamespace())

    assert "Completed preparation step." in caplog.text
