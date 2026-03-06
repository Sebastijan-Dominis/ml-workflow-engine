"""Unit tests for search metadata preparation helper."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from ml.search.persistence import prepare_metadata as prepare_metadata_module

pytestmark = pytest.mark.unit


class _RecordStub:
    """Validation-result stub that tracks ``model_dump`` call options."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.model_dump_kwargs: dict[str, Any] | None = None

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        """Return captured payload and store kwargs for assertion."""
        self.model_dump_kwargs = kwargs
        return self.payload


def _build_base_model_cfg(*, task_type: str) -> Any:
    """Create minimal config-like object matching helper attribute access."""
    return SimpleNamespace(
        search_lineage=SimpleNamespace(created_at="search-created-at"),
        model_specs_lineage=SimpleNamespace(created_at="specs-created-at"),
        model_dump=lambda **kwargs: {
            "search_lineage": {"created_at": "raw-search-created-at"},
            "model_specs_lineage": {"created_at": "raw-specs-created-at"},
            "dump_kwargs": kwargs,
        },
        problem="cancellation",
        segment=SimpleNamespace(name="city_hotel"),
        version="v1",
        algorithm=SimpleNamespace(value="catboost"),
        seed=None,
        search=SimpleNamespace(hardware={"task_type": "CPU"}),
        meta=SimpleNamespace(
            sources=None,
            env=None,
            best_params_path=None,
            config_hash=None,
            validation_status=None,
        ),
        pipeline=SimpleNamespace(version="pipe-v2"),
        task=SimpleNamespace(type=task_type),
        target=SimpleNamespace(
            transform=SimpleNamespace(enabled=True, type="log1p", lambda_value=0.5)
        ),
        class_weighting=SimpleNamespace(model_dump=lambda: {"policy": "balanced"}),
    )


def test_prepare_metadata_classification_builds_record_with_defaults_and_class_weighting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Build classification metadata with default fallbacks and validate payload wiring."""
    model_cfg = _build_base_model_cfg(task_type="classification")
    feature_lineage = [
        SimpleNamespace(model_dump=lambda: {"name": "f1"}),
        SimpleNamespace(model_dump=lambda: {"name": "f2"}),
    ]
    captured_raw: dict[str, Any] = {}
    record_stub = _RecordStub(payload={"status": "ok", "metadata": {"kind": "classification"}})

    monkeypatch.setattr(
        prepare_metadata_module,
        "iso_no_colon",
        lambda value: f"iso::{value}",
    )
    monkeypatch.setattr(prepare_metadata_module, "get_git_commit", lambda path: "commit-abc" if path == Path(".") else "unexpected")
    monkeypatch.setattr(prepare_metadata_module, "asdict", lambda splits: {"train_rows": 100, "val_rows": 25})

    def _validate(raw: dict[str, Any]) -> _RecordStub:
        captured_raw.update(raw)
        return record_stub

    monkeypatch.setattr(prepare_metadata_module, "validate_search_record", _validate)

    result = prepare_metadata_module.prepare_metadata(
        cast(Any, model_cfg),
        search_results={"best": {"score": 0.81}},
        owner="alice",
        experiment_id="exp-1",
        timestamp="20260306T100000",
        feature_lineage=cast(Any, feature_lineage),
        pipeline_hash="pipe-hash-1",
        scoring_method="roc_auc",
        splits_info=cast(Any, object()),
    )

    assert result == {"status": "ok", "metadata": {"kind": "classification"}}
    assert record_stub.model_dump_kwargs == {"exclude_none": True}

    assert captured_raw["metadata"]["seed"] == "none"
    assert captured_raw["metadata"]["sources"] == {}
    assert captured_raw["metadata"]["env"] == "default"
    assert captured_raw["metadata"]["best_params_path"] == "none"
    assert captured_raw["metadata"]["config_hash"] == "none"
    assert captured_raw["metadata"]["validation_status"] == "unknown"
    assert captured_raw["metadata"]["git_commit"] == "commit-abc"
    assert captured_raw["metadata"]["feature_lineage"] == [{"name": "f1"}, {"name": "f2"}]
    assert captured_raw["metadata"]["class_weighting"] == {"policy": "balanced"}
    assert "target_transform" not in captured_raw["metadata"]
    assert captured_raw["metadata"]["splits_info"] == {"train_rows": 100, "val_rows": 25}
    assert captured_raw["config"]["search_lineage"]["created_at"] == "iso::search-created-at"
    assert captured_raw["config"]["model_specs_lineage"]["created_at"] == "iso::specs-created-at"
    assert captured_raw["config"]["dump_kwargs"] == {"by_alias": True}


def test_prepare_metadata_regression_adds_target_transform_and_omits_none_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Include regression target-transform metadata while dropping optional None fields."""
    model_cfg = _build_base_model_cfg(task_type="regression")
    model_cfg.target = SimpleNamespace(
        transform=SimpleNamespace(enabled=False, type=None, lambda_value=None)
    )
    model_cfg.seed = 77
    model_cfg.meta = SimpleNamespace(
        sources={"cfg": "path.yaml"},
        env="prod",
        best_params_path="params/best.json",
        config_hash="cfg-hash-1",
        validation_status="ok",
    )
    captured_raw: dict[str, Any] = {}

    monkeypatch.setattr(prepare_metadata_module, "iso_no_colon", lambda value: str(value))
    monkeypatch.setattr(prepare_metadata_module, "get_git_commit", lambda path: "commit-reg")
    monkeypatch.setattr(prepare_metadata_module, "asdict", lambda splits: {"train_rows": 10})
    monkeypatch.setattr(
        prepare_metadata_module,
        "validate_search_record",
        lambda raw: captured_raw.update(raw) or _RecordStub(payload={"ok": True}),
    )

    result = prepare_metadata_module.prepare_metadata(
        cast(Any, model_cfg),
        search_results={"best": {"rmse": 8.2}},
        owner="bob",
        experiment_id="exp-2",
        timestamp="20260306T100001",
        feature_lineage=cast(Any, []),
        pipeline_hash="pipe-hash-2",
        scoring_method="neg_root_mean_squared_error",
        splits_info=cast(Any, object()),
    )

    assert result == {"ok": True}
    assert captured_raw["metadata"]["seed"] == 77
    assert captured_raw["metadata"]["sources"] == {"cfg": "path.yaml"}
    assert captured_raw["metadata"]["env"] == "prod"
    assert captured_raw["metadata"]["best_params_path"] == "params/best.json"
    assert captured_raw["metadata"]["config_hash"] == "cfg-hash-1"
    assert captured_raw["metadata"]["validation_status"] == "ok"
    assert captured_raw["metadata"]["target_transform"] == {"enabled": False}
    assert "class_weighting" not in captured_raw["metadata"]
