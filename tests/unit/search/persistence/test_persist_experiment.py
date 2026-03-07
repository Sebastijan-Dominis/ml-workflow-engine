"""Unit tests for search experiment persistence orchestration helper."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from ml.search.persistence import persist_experiment as module

pytestmark = pytest.mark.unit


def _build_model_cfg() -> Any:
    """Create a minimal config-like object for persistence call wiring tests."""
    return SimpleNamespace(search=SimpleNamespace(hardware={"task_type": "CPU", "devices": "0"}))


def test_persist_experiment_prepares_metadata_and_persists_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Prepare metadata once, then persist metadata and runtime snapshot with consistent payloads."""
    model_cfg = _build_model_cfg()
    search_dir = tmp_path / "experiments" / "no_show" / "global" / "v1" / "exp_10" / "search"

    captured_prepare: dict[str, Any] = {}
    captured_metadata: dict[str, Any] = {}
    captured_runtime: dict[str, Any] = {}

    def _prepare_metadata(cfg: Any, **kwargs: Any) -> dict[str, Any]:
        captured_prepare["cfg"] = cfg
        captured_prepare.update(kwargs)
        return {"metadata": {"experiment_id": kwargs["experiment_id"]}}

    def _save_metadata(**kwargs: Any) -> None:
        captured_metadata.update(kwargs)

    def _save_runtime_snapshot(**kwargs: Any) -> None:
        captured_runtime.update(kwargs)

    monkeypatch.setattr(module, "prepare_metadata", _prepare_metadata)
    monkeypatch.setattr(module, "save_metadata", _save_metadata)
    monkeypatch.setattr(module, "save_runtime_snapshot", _save_runtime_snapshot)

    feature_lineage = [SimpleNamespace(model_dump=lambda: {"name": "feature_a"})]
    splits_info = cast(Any, {"train_rows": 120, "val_rows": 30, "test_rows": 30})

    module.persist_experiment(
        cast(Any, model_cfg),
        search_results={"best": {"score": 0.81}},
        owner="CI",
        experiment_id="exp_10",
        search_dir=search_dir,
        timestamp="20260307T140000",
        start_time=1234.56,
        feature_lineage=cast(Any, feature_lineage),
        pipeline_hash="pipeline-hash-123",
        scoring_method="roc_auc",
        splits_info=splits_info,
    )

    assert captured_prepare["cfg"] is model_cfg
    assert captured_prepare["search_results"] == {"best": {"score": 0.81}}
    assert captured_prepare["owner"] == "CI"
    assert captured_prepare["experiment_id"] == "exp_10"
    assert captured_prepare["timestamp"] == "20260307T140000"
    assert captured_prepare["feature_lineage"] == feature_lineage
    assert captured_prepare["pipeline_hash"] == "pipeline-hash-123"
    assert captured_prepare["scoring_method"] == "roc_auc"
    assert captured_prepare["splits_info"] == splits_info

    assert captured_metadata == {
        "metadata": {"metadata": {"experiment_id": "exp_10"}},
        "target_dir": search_dir,
        "overwrite_existing": False,
    }

    assert captured_runtime == {
        "target_dir": search_dir,
        "timestamp": "20260307T140000",
        "hardware_info": {"task_type": "CPU", "devices": "0"},
        "start_time": 1234.56,
        "overwrite_existing": False,
    }


def test_persist_experiment_propagates_overwrite_flag_to_all_persistence_steps(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Forward overwrite intent consistently to metadata and runtime persistence backends."""
    model_cfg = _build_model_cfg()
    search_dir = tmp_path / "search"

    monkeypatch.setattr(module, "prepare_metadata", lambda *_args, **_kwargs: {"ok": True})

    metadata_flags: list[bool] = []
    runtime_flags: list[bool] = []

    monkeypatch.setattr(
        module,
        "save_metadata",
        lambda **kwargs: metadata_flags.append(cast(bool, kwargs["overwrite_existing"])),
    )
    monkeypatch.setattr(
        module,
        "save_runtime_snapshot",
        lambda **kwargs: runtime_flags.append(cast(bool, kwargs["overwrite_existing"])),
    )

    module.persist_experiment(
        cast(Any, model_cfg),
        search_results={},
        owner="owner",
        experiment_id="exp_overwrite",
        search_dir=search_dir,
        timestamp="20260307T141500",
        start_time=1.23,
        feature_lineage=cast(Any, []),
        pipeline_hash="hash",
        scoring_method="roc_auc",
        splits_info=cast(Any, {}),
        overwrite_existing=True,
    )

    assert metadata_flags == [True]
    assert runtime_flags == [True]
