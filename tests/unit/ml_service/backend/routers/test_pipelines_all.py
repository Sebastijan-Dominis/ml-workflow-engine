"""Tests exercising all endpoints in `ml_service.backend.routers.pipelines`.

These tests bypass the rate-limiter decorator by invoking the wrapped
function and monkeypatch `execute_pipeline` to capture calls.
"""

from __future__ import annotations

from fastapi import Request


def test_all_pipeline_endpoints_call_execute_pipeline(monkeypatch):
    import ml_service.backend.routers.pipelines as pl_mod

    calls = []

    def fake_execute_pipeline(module_path, payload, boolean_args):
        calls.append((module_path, payload, boolean_args))
        return {"module": module_path}

    monkeypatch.setattr("ml_service.backend.routers.pipelines.execute_pipeline", fake_execute_pipeline)

    req = Request({"type": "http", "method": "POST", "path": "/", "headers": []})

    mapping = [
        ("register_raw_snapshot", "pipelines.data.register_raw_snapshot"),
        ("build_interim_dataset", "pipelines.data.build_interim_dataset"),
        ("build_processed_dataset", "pipelines.data.build_processed_dataset"),
        ("freeze_feature_set", "pipelines.features.freeze"),
        ("search", "pipelines.search.search"),
        ("train", "pipelines.runners.train"),
        ("evaluate", "pipelines.runners.evaluate"),
        ("explain", "pipelines.runners.explain"),
        ("promote", "pipelines.promotion.promote"),
        ("execute_all_data_preprocessing", "pipelines.orchestration.data.execute_all_data_preprocessing"),
        ("freeze_all_feature_sets", "pipelines.orchestration.features.freeze_all_feature_sets"),
        ("execute_experiment_with_latest", "pipelines.orchestration.experiments.execute_experiment_with_latest"),
        ("execute_all_experiments_with_latest", "pipelines.orchestration.experiments.execute_all_experiments_with_latest"),
        ("run_all_workflows", "pipelines.orchestration.master.run_all_workflows"),
        ("infer", "pipelines.post_promotion.infer"),
        ("monitor", "pipelines.post_promotion.monitor"),
    ]

    for func_name, expected_module in mapping:
        func = getattr(pl_mod, func_name)
        orig = getattr(func, "__wrapped__", func)
        res = orig({}, req)
        assert res["module"] == expected_module
