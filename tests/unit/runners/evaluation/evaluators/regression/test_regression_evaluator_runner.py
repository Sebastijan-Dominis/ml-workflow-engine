"""Unit tests for the ``EvaluateRegression`` runner orchestration."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest
from ml.config.schemas.model_cfg import TrainModelConfig
from ml.exceptions import PipelineContractError
from ml.runners.evaluation.constants.output import EvaluateOutput

pytestmark = pytest.mark.unit


def _get_evaluate_regression() -> type[Any]:
    """Import ``EvaluateRegression`` with lightweight registry stubs to avoid cycles."""
    if "ml.registries" not in sys.modules:
        registries_stub = ModuleType("ml.registries")
        registries_stub.__path__ = []  # type: ignore[attr-defined]
        sys.modules["ml.registries"] = registries_stub

    catalogs_stub = ModuleType("ml.registries.catalogs")
    catalogs_stub.__dict__["OP_MAP"] = {}
    sys.modules["ml.registries.catalogs"] = catalogs_stub

    module = importlib.import_module("ml.runners.evaluation.evaluators.regression.regression")
    return cast(type[Any], module.EvaluateRegression)


def test_evaluate_raises_when_training_metadata_lacks_pipeline_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Raise ``PipelineContractError`` when training metadata omits pipeline path."""
    monkeypatch.setattr(
        "ml.runners.evaluation.evaluators.regression.regression.load_json",
        lambda _path: {"artifacts": {}},
    )
    monkeypatch.setattr(
        "ml.runners.evaluation.evaluators.regression.regression.validate_training_metadata",
        lambda _raw: SimpleNamespace(artifacts=SimpleNamespace(pipeline_path=None)),
    )

    evaluator = _get_evaluate_regression()()

    with pytest.raises(PipelineContractError, match="missing the path to the trained pipeline"):
        evaluator.evaluate(
            model_cfg=cast(TrainModelConfig, SimpleNamespace()),
            strict=True,
            best_threshold=None,
            train_dir=tmp_path,
        )


def test_evaluate_raises_when_loaded_pipeline_has_no_predict(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Raise ``PipelineContractError`` when loaded pipeline violates predict contract."""
    monkeypatch.setattr(
        "ml.runners.evaluation.evaluators.regression.regression.load_json",
        lambda _path: {"artifacts": {"pipeline_path": str(tmp_path / "pipeline.pkl")}},
    )
    monkeypatch.setattr(
        "ml.runners.evaluation.evaluators.regression.regression.validate_training_metadata",
        lambda _raw: SimpleNamespace(artifacts=SimpleNamespace(pipeline_path=str(tmp_path / "pipeline.pkl"))),
    )
    monkeypatch.setattr(
        "ml.runners.evaluation.evaluators.regression.regression.load_model_or_pipeline",
        lambda _path, _kind: object(),
    )

    evaluator = _get_evaluate_regression()()

    with pytest.raises(PipelineContractError, match="does not implement 'predict'"):
        evaluator.evaluate(
            model_cfg=cast(TrainModelConfig, SimpleNamespace()),
            strict=False,
            best_threshold=None,
            train_dir=tmp_path,
        )


def test_evaluate_loads_artifacts_and_returns_evaluate_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Orchestrate pipeline loading, split creation, and output assembly correctly."""

    class _Pipeline:
        """Minimal pipeline test double exposing ``predict`` method."""

        def predict(self, X: pd.DataFrame) -> pd.Series:
            """Return deterministic values for compatibility with evaluator contracts."""
            return pd.Series([0.0] * len(X))

    pipeline_path = tmp_path / "pipeline.pkl"
    training_metadata = SimpleNamespace(artifacts=SimpleNamespace(pipeline_path=str(pipeline_path)))

    monkeypatch.setattr(
        "ml.runners.evaluation.evaluators.regression.regression.load_json",
        lambda _path: {"artifacts": {"pipeline_path": str(pipeline_path)}},
    )
    monkeypatch.setattr(
        "ml.runners.evaluation.evaluators.regression.regression.validate_training_metadata",
        lambda _raw: training_metadata,
    )
    monkeypatch.setattr(
        "ml.runners.evaluation.evaluators.regression.regression.load_model_or_pipeline",
        lambda _path, _kind: _Pipeline(),
    )
    monkeypatch.setattr(
        "ml.runners.evaluation.evaluators.regression.regression.get_snapshot_binding_from_training_metadata",
        lambda _meta: {"booking_context_features": "2026-03-07"},
    )

    snapshot_selection = {"booking_context_features": "2026-03-07"}
    monkeypatch.setattr(
        "ml.runners.evaluation.evaluators.regression.regression.resolve_feature_snapshots",
        lambda **_kwargs: snapshot_selection,
    )

    X = pd.DataFrame({"row_id": ["r1", "r2", "r3"], "feature": [0.1, 0.2, 0.3]})
    y = pd.Series([1.0, 2.0, 3.0], name="target")
    lineage = [SimpleNamespace(feature_set="booking_context_features")]
    monkeypatch.setattr(
        "ml.runners.evaluation.evaluators.regression.regression.load_features_and_target",
        lambda *_args, **_kwargs: (X, y, lineage, "entity_key"),
    )

    validate_snapshot_ids_calls: list[tuple[Any, Any]] = []

    def _validate_snapshot_ids(feature_lineage: Any, selected: Any) -> None:
        validate_snapshot_ids_calls.append((feature_lineage, selected))

    monkeypatch.setattr(
        "ml.runners.evaluation.evaluators.regression.regression.validate_snapshot_ids",
        _validate_snapshot_ids,
    )

    splits = SimpleNamespace(
        X_train=X.iloc[0:1].copy(),
        y_train=y.iloc[0:1].copy(),
        X_val=X.iloc[1:2].copy(),
        y_val=y.iloc[1:2].copy(),
        X_test=X.iloc[2:3].copy(),
        y_test=y.iloc[2:3].copy(),
    )
    monkeypatch.setattr(
        "ml.runners.evaluation.evaluators.regression.regression.get_splits",
        lambda **_kwargs: (splits, {"summary": "ok"}),
    )

    expected_metrics = {"train": {"rmse": 0.0}, "val": {"rmse": 0.0}, "test": {"rmse": 0.0}}
    expected_prediction_artifacts = SimpleNamespace(train=pd.DataFrame(), val=pd.DataFrame(), test=pd.DataFrame())
    evaluate_model_calls: list[dict[str, Any]] = []

    def _evaluate_model(**kwargs: Any) -> tuple[dict[str, dict[str, float]], Any]:
        evaluate_model_calls.append(kwargs)
        return expected_metrics, expected_prediction_artifacts

    monkeypatch.setattr(
        "ml.runners.evaluation.evaluators.regression.regression.evaluate_model",
        _evaluate_model,
    )

    model_cfg = cast(
        TrainModelConfig,
        SimpleNamespace(
            feature_store=SimpleNamespace(path=str(tmp_path), feature_sets=["booking_context_features"]),
            split=SimpleNamespace(type="random", train_size=0.7),
            data_type="batch",
            task=SimpleNamespace(type="regression", subtype=None),
            target=SimpleNamespace(transform=SimpleNamespace(enabled=False, type=None, lambda_value=None)),
        ),
    )

    evaluator = _get_evaluate_regression()()
    output = evaluator.evaluate(
        model_cfg=model_cfg,
        strict=True,
        best_threshold=None,
        train_dir=tmp_path,
    )

    assert isinstance(output, EvaluateOutput)
    assert output.metrics == expected_metrics
    assert output.prediction_dfs is expected_prediction_artifacts
    assert output.lineage == lineage

    assert validate_snapshot_ids_calls == [(lineage, snapshot_selection)]
    assert len(evaluate_model_calls) == 1
    call_kwargs = evaluate_model_calls[0]
    assert hasattr(call_kwargs["pipeline"], "predict")
    assert call_kwargs["transform_cfg"] == model_cfg.target.transform

    data_splits = call_kwargs["data_splits"]
    assert data_splits.train[0].equals(splits.X_train)
    assert data_splits.val[0].equals(splits.X_val)
    assert data_splits.test[0].equals(splits.X_test)
