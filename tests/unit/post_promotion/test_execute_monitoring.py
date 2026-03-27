import importlib
from types import SimpleNamespace

import pandas as pd


def test_execute_monitoring_monkeypatched(monkeypatch):
    exec_mod = importlib.import_module("ml.post_promotion.monitoring.execution.execute_monitoring")

    args = SimpleNamespace(problem="p", segment="s")

    training_features = pd.DataFrame({"a": [1, 2]})
    inference_features = pd.DataFrame({"a": [1, 2]})
    inference_return = SimpleNamespace(features=inference_features, target=pd.Series([0, 1]))

    monkeypatch.setattr(exec_mod, "load_training_features", lambda *a, **kw: training_features)
    monkeypatch.setattr(exec_mod, "load_inference_features_and_target", lambda *a, **kw: inference_return)
    monkeypatch.setattr(exec_mod, "compare_feature_distributions", lambda *a, **kw: {"a": 0.0})
    monkeypatch.setattr(exec_mod, "assess_model_performance", lambda *a, **kw: {"metric": {"value": 1.0}})

    from ml.post_promotion.monitoring.classes.function_returns import MonitoringExecutionOutput
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

    out = exec_mod.execute_monitoring(args=args, model_metadata=reg, stage="production", promotion_metrics_info=None)

    assert isinstance(out, MonitoringExecutionOutput)
    assert out.model_version == "mv"
    assert out.drift_results == {"a": 0.0}
