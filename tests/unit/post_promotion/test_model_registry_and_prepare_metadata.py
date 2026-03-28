import argparse
from datetime import datetime

import ml.post_promotion.monitoring.persistence.prepare_metadata as pm_mod
import ml.post_promotion.shared.loading.model_registry as mr_mod
import pytest
from ml.exceptions import PipelineContractError
from ml.post_promotion.monitoring.classes.function_returns import MonitoringExecutionOutput


def test_get_model_registry_info_raises_when_missing(monkeypatch):
    monkeypatch.setattr(mr_mod, "load_yaml", lambda path: {})
    args = argparse.Namespace(problem="p", segment="s")

    with pytest.raises(PipelineContractError):
        mr_mod.get_model_registry_info(args)


def test_get_model_registry_info_returns_validated_entries(monkeypatch):
    raw = {"p": {"s": {"production": {"a": 1}, "staging": {"b": 2}}}}
    monkeypatch.setattr(mr_mod, "load_yaml", lambda path: raw)
    monkeypatch.setattr(mr_mod, "validate_registry_entry", lambda x: f"VALID:{x}")

    args = argparse.Namespace(problem="p", segment="s")
    res = mr_mod.get_model_registry_info(args)

    assert res.prod_meta == "VALID:{'a': 1}"
    assert res.stage_meta == "VALID:{'b': 2}"


def test_prepare_metadata_without_outputs():
    args = argparse.Namespace(problem="prob", segment="seg", inference_run_id="ir")
    run_id = "run123"
    timestamp = datetime(2026, 3, 27, 12, 0)

    res = pm_mod.prepare_metadata(args=args, run_id=run_id, timestamp=timestamp)

    assert res["problem_type"] == "prob"
    assert res["segment"] == "seg"
    assert res["timestamp"] == timestamp.isoformat()
    assert res["run_id"] == run_id
    assert res["inference_run_id"] == "ir"
    assert "production" not in res
    assert "staging" not in res


def test_prepare_metadata_with_outputs():
    args = argparse.Namespace(problem="prob", segment="seg", inference_run_id="ir")
    run_id = "run123"
    timestamp = datetime(2026, 3, 27, 12, 0)

    prod = MonitoringExecutionOutput(drift_results={"d": 1.0}, performance_results={"m": {"current": 0.5, "direction": "maximize"}}, model_version="v1")
    stage = MonitoringExecutionOutput(drift_results={"d": 0.5}, performance_results={"m": {"current": 0.6, "direction": "maximize"}}, model_version="v2")

    res = pm_mod.prepare_metadata(args=args, run_id=run_id, timestamp=timestamp, prod_monitoring_output=prod, stage_monitoring_output=stage)

    assert res["production"] == prod.__dict__
    assert res["staging"] == stage.__dict__
