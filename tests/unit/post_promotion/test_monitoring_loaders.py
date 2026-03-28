import argparse
from pathlib import Path

import ml.post_promotion.monitoring.loading.inference_features_and_target as ift_mod
import ml.post_promotion.monitoring.loading.predictions as pred_mod
import ml.post_promotion.monitoring.loading.promotion_metrics_info as pmi_mod
import ml.post_promotion.monitoring.loading.training_features as tf_mod
import ml.post_promotion.monitoring.loading.training_metrics as tm_mod
import pandas as pd
import pytest
from ml.exceptions import PipelineContractError
from ml.modeling.models.metrics import TrainingMetrics
from ml.promotion.config.registry_entry import (
    RegistryArtifacts,
    RegistryEntry,
    RegistryEntryMetrics,
    RegistryFeatureSetLineage,
)


def test_load_predictions_returns_dataframe(monkeypatch, tmp_path):
    args = argparse.Namespace(inference_run_id="run1", problem="prob", segment="seg")
    df = pd.DataFrame({"a": [1, 2]})
    monkeypatch.setattr(pred_mod, "get_snapshot_path", lambda run_id, inference_dir: Path(tmp_path))
    monkeypatch.setattr(pred_mod.pd, "read_parquet", lambda path: df)

    res = pred_mod.load_predictions(args, "production")
    pd.testing.assert_frame_equal(res, df)


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


def test_load_inference_features_and_target(monkeypatch):
    args = argparse.Namespace(inference_run_id="run1", problem="p", segment="s")
    model_metadata = _build_registry_entry()

    monkeypatch.setattr(ift_mod, "get_snapshot_path", lambda run_id, inference_dir: Path("dummy"))
    monkeypatch.setattr(ift_mod, "load_json", lambda path: {"snapshot_bindings_id": "sb1"})
    monkeypatch.setattr(ift_mod, "validate_inference_metadata", lambda raw: argparse.Namespace(snapshot_bindings_id="sb1"))

    features_df = pd.DataFrame({"f": [1]})
    target_s = pd.Series([0], name="target")

    def fake_prepare_features(*_args, **_kwargs):
        return argparse.Namespace(features=features_df, target=target_s)

    monkeypatch.setattr(ift_mod, "prepare_features", fake_prepare_features)

    res = ift_mod.load_inference_features_and_target(args, model_metadata, "production")
    pd.testing.assert_frame_equal(res.features, features_df)
    pd.testing.assert_series_equal(res.target, target_s)


def test_load_training_features(monkeypatch):
    args = argparse.Namespace()
    model_metadata = _build_registry_entry()
    features_df = pd.DataFrame({"x": [1]})
    monkeypatch.setattr(tf_mod, "prepare_features", lambda *a, **k: argparse.Namespace(features=features_df))

    res = tf_mod.load_training_features(args=args, model_metadata=model_metadata)
    pd.testing.assert_frame_equal(res, features_df)


def test_load_training_metrics_file(monkeypatch):
    args = argparse.Namespace(problem="p", segment="s")
    model_metadata = _build_registry_entry()
    training_metrics_raw = {"task_type": "classification", "algorithm": "xgb", "metrics": {"acc": 0.9}}
    monkeypatch.setattr(tm_mod, "load_json", lambda path: training_metrics_raw)

    res = tm_mod.load_training_metrics_file(args, model_metadata)
    assert isinstance(res, TrainingMetrics)
    assert res.metrics == training_metrics_raw["metrics"]


def test_get_promotion_metrics_info_raises_when_missing(monkeypatch):
    args = argparse.Namespace(problem="p", segment="s")
    monkeypatch.setattr(pmi_mod, "load_yaml", lambda path: {})

    with pytest.raises(PipelineContractError):
        pmi_mod.get_promotion_metrics_info(args)


def test_get_promotion_metrics_info_returns(monkeypatch):
    args = argparse.Namespace(problem="p", segment="s")
    raw = {"p": {"s": {"foo": "bar"}}}
    monkeypatch.setattr(pmi_mod, "load_yaml", lambda path: raw)
    monkeypatch.setattr(pmi_mod, "validate_promotion_thresholds", lambda x: argparse.Namespace(promotion_metrics="pm"))

    res = pmi_mod.get_promotion_metrics_info(args)
    assert res == "pm"
