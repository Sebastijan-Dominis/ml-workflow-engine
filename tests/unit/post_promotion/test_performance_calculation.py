import argparse
from typing import Any

import ml.post_promotion.monitoring.performance.calculation as calc_mod
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


def test_classification_with_threshold(monkeypatch):
    args = argparse.Namespace()
    model_metadata = _build_registry_entry()

    predictions = pd.DataFrame({"prediction": [1, 0], "proba_1": [0.7, 0.2]}, index=[0, 1])
    monkeypatch.setattr(calc_mod, "load_predictions", lambda a, s: predictions)

    training_raw = {"task_type": "classification", "algorithm": "x", "metrics": {"threshold": {"value": 0.6}}}
    training_metrics = TrainingMetrics(**training_raw)
    monkeypatch.setattr(calc_mod, "load_training_metrics_file", lambda a, m: training_metrics)

    called: dict[str, Any] = {}

    def fake_compute(y_true, y_pred, y_prob, threshold):
        called["threshold"] = threshold
        assert y_prob is not None
        return {"acc": 0.9}

    monkeypatch.setattr(calc_mod, "compute_classification_metrics", fake_compute)

    target = pd.Series([1, 0], index=[0, 1])
    res = calc_mod.calculate_current_performance(args=args, model_metadata=model_metadata, stage="production", target=target)

    assert called["threshold"] == 0.6
    assert res["acc"] == 0.9


def test_classification_missing_threshold_uses_default(monkeypatch):
    args = argparse.Namespace()
    model_metadata = _build_registry_entry()

    predictions = pd.DataFrame({"prediction": [0, 1]}, index=[0, 1])
    monkeypatch.setattr(calc_mod, "load_predictions", lambda a, s: predictions)

    training_raw = {"task_type": "classification", "algorithm": "x", "metrics": {}}
    training_metrics = TrainingMetrics(**training_raw)
    monkeypatch.setattr(calc_mod, "load_training_metrics_file", lambda a, m: training_metrics)

    called: dict[str, Any] = {}

    def fake_compute(y_true, y_pred, y_prob, threshold):
        called["threshold"] = threshold
        return {"acc": 0.1}

    monkeypatch.setattr(calc_mod, "compute_classification_metrics", fake_compute)

    target = pd.Series([0, 1], index=[0, 1])
    res = calc_mod.calculate_current_performance(args=args, model_metadata=model_metadata, stage="production", target=target)

    assert called["threshold"] == 0.5
    assert res["acc"] == 0.1


def test_regression_calls_regression_metrics(monkeypatch):
    args = argparse.Namespace()
    model_metadata = _build_registry_entry()

    predictions = pd.DataFrame({"prediction": [1.2, 0.7]}, index=[0, 1])
    monkeypatch.setattr(calc_mod, "load_predictions", lambda a, s: predictions)

    training_raw = {"task_type": "regression", "algorithm": "rf", "metrics": {}}
    training_metrics = TrainingMetrics(**training_raw)
    monkeypatch.setattr(calc_mod, "load_training_metrics_file", lambda a, m: training_metrics)

    called: dict[str, Any] = {}

    def fake_regression(y_true, y_pred):
        called["called"] = True
        return {"rmse": 0.123}

    monkeypatch.setattr(calc_mod, "compute_regression_metrics", fake_regression)

    target = pd.Series([1.0, 1.0], index=[0, 1])
    res = calc_mod.calculate_current_performance(args=args, model_metadata=model_metadata, stage="production", target=target)
    assert called.get("called") is True
    assert res["rmse"] == 0.123


def test_unsupported_task_type_raises(monkeypatch):
    args = argparse.Namespace()
    model_metadata = _build_registry_entry()

    predictions = pd.DataFrame({"prediction": [1, 0]}, index=[0, 1])
    monkeypatch.setattr(calc_mod, "load_predictions", lambda a, s: predictions)

    training_raw = {"task_type": "weird", "algorithm": "x", "metrics": {}}
    training_metrics = TrainingMetrics(**training_raw)
    monkeypatch.setattr(calc_mod, "load_training_metrics_file", lambda a, m: training_metrics)

    target = pd.Series([1, 0], index=[0, 1])
    with pytest.raises(PipelineContractError):
        calc_mod.calculate_current_performance(args=args, model_metadata=model_metadata, stage="production", target=target)
