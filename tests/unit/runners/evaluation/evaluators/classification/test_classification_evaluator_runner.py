"""Unit tests for ``ClassificationEvaluator`` orchestration behavior."""

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


def _get_classification_evaluator() -> type[Any]:
    """Import ``ClassificationEvaluator`` with lightweight registry stubs."""
    if "ml.registries" not in sys.modules:
        registries_stub = ModuleType("ml.registries")
        registries_stub.__path__ = []  # type: ignore[attr-defined]
        sys.modules["ml.registries"] = registries_stub

    catalogs_stub = ModuleType("ml.registries.catalogs")
    catalogs_stub.__dict__["OP_MAP"] = {}
    sys.modules["ml.registries.catalogs"] = catalogs_stub

    module = importlib.import_module("ml.runners.evaluation.evaluators.classification.classification")
    return cast(type[Any], module.ClassificationEvaluator)


def _get_classification_module() -> ModuleType:
    """Return imported classification evaluator module with cycle-safe stubs set."""
    _get_classification_evaluator()
    return cast(ModuleType, sys.modules["ml.runners.evaluation.evaluators.classification.classification"])


def test_evaluate_warns_and_defaults_threshold_for_binary_when_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Warn and default to 0.5 threshold when binary threshold is omitted."""
    module = _get_classification_module()
    pipeline_path = tmp_path / "pipeline.pkl"
    training_metadata = SimpleNamespace(artifacts=SimpleNamespace(pipeline_path=str(pipeline_path)))

    monkeypatch.setattr(
        module,
        "load_json",
        lambda _path: {"artifacts": {"pipeline_path": str(pipeline_path)}},
    )
    monkeypatch.setattr(
        module,
        "validate_training_metadata",
        lambda _raw: training_metadata,
    )

    class _Pipeline:
        """Minimal classifier test double exposing ``predict_proba``."""

        def predict_proba(self, X: pd.DataFrame) -> list[list[float]]:
            """Return deterministic probabilities sized to the input dataframe."""
            return [[0.4, 0.6] for _ in range(len(X))]

    monkeypatch.setattr(
        module,
        "load_model_or_pipeline",
        lambda _path, _kind: _Pipeline(),
    )
    monkeypatch.setattr(
        module,
        "get_snapshot_binding_from_training_metadata",
        lambda _meta: {"booking_context_features": "2026-03-07"},
    )
    monkeypatch.setattr(
        module,
        "resolve_feature_snapshots",
        lambda **_kwargs: {"booking_context_features": "2026-03-07"},
    )

    X = pd.DataFrame({"row_id": ["r1", "r2"], "feature": [0.1, 0.2]})
    y = pd.Series([0, 1], name="target")
    lineage = [SimpleNamespace(feature_set="booking_context_features")]
    monkeypatch.setattr(
        module,
        "load_features_and_target",
        lambda *_args, **_kwargs: (X, y, lineage, "entity_key"),
    )
    monkeypatch.setattr(
        module,
        "validate_snapshot_ids",
        lambda *_args, **_kwargs: None,
    )

    splits = SimpleNamespace(
        X_train=X.iloc[0:1].copy(),
        y_train=y.iloc[0:1].copy(),
        X_val=X.iloc[1:2].copy(),
        y_val=y.iloc[1:2].copy(),
        X_test=X.iloc[0:1].copy(),
        y_test=y.iloc[0:1].copy(),
    )
    monkeypatch.setattr(
        module,
        "get_splits",
        lambda **_kwargs: (splits, {"summary": "ok"}),
    )

    evaluate_calls: list[dict[str, Any]] = []

    def _evaluate_model(*args: Any, **kwargs: Any) -> tuple[dict[str, dict[str, float]], Any]:
        evaluate_calls.append({"args": args, "kwargs": kwargs})
        return (
            {"train": {"accuracy": 1.0}, "val": {"accuracy": 1.0}, "test": {"accuracy": 1.0}},
            SimpleNamespace(train=pd.DataFrame(), val=pd.DataFrame(), test=pd.DataFrame()),
        )

    monkeypatch.setattr(
        module,
        "evaluate_model",
        _evaluate_model,
    )

    model_cfg = cast(
        TrainModelConfig,
        SimpleNamespace(
            feature_store=SimpleNamespace(path=str(tmp_path), feature_sets=["booking_context_features"]),
            split=SimpleNamespace(type="random", train_size=0.7),
            data_type="batch",
            task=SimpleNamespace(type="classification", subtype="binary"),
            target=SimpleNamespace(transform=SimpleNamespace(enabled=False, type=None, lambda_value=None)),
        ),
    )

    evaluator = _get_classification_evaluator()()
    with caplog.at_level("WARNING", logger="ml.runners.evaluation.evaluators.classification.classification"):
        output = evaluator.evaluate(
            model_cfg=model_cfg,
            strict=True,
            best_threshold=None,
            train_dir=tmp_path,
        )

    assert isinstance(output, EvaluateOutput)
    assert len(evaluate_calls) == 1
    assert evaluate_calls[0]["kwargs"]["best_threshold"] == 0.5
    assert "Defaulting to 0.5" in caplog.text


def test_evaluate_raises_when_training_metadata_lacks_pipeline_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raise ``PipelineContractError`` when pipeline artifact path is missing."""
    module = _get_classification_module()
    monkeypatch.setattr(
        module,
        "load_json",
        lambda _path: {"artifacts": {}},
    )
    monkeypatch.setattr(
        module,
        "validate_training_metadata",
        lambda _raw: SimpleNamespace(artifacts=SimpleNamespace(pipeline_path=None)),
    )

    evaluator = _get_classification_evaluator()()

    with pytest.raises(PipelineContractError, match="missing the path to the trained pipeline"):
        evaluator.evaluate(
            model_cfg=cast(TrainModelConfig, SimpleNamespace(task=SimpleNamespace(type="classification", subtype="binary"))),
            strict=True,
            best_threshold=0.5,
            train_dir=tmp_path,
        )


def test_evaluate_raises_when_loaded_pipeline_lacks_predict_proba(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raise ``PipelineContractError`` when loaded pipeline lacks ``predict_proba``."""
    module = _get_classification_module()
    monkeypatch.setattr(
        module,
        "load_json",
        lambda _path: {"artifacts": {"pipeline_path": str(tmp_path / "pipeline.pkl")}},
    )
    monkeypatch.setattr(
        module,
        "validate_training_metadata",
        lambda _raw: SimpleNamespace(artifacts=SimpleNamespace(pipeline_path=str(tmp_path / "pipeline.pkl"))),
    )
    monkeypatch.setattr(
        module,
        "load_model_or_pipeline",
        lambda _path, _kind: object(),
    )

    evaluator = _get_classification_evaluator()()

    with pytest.raises(PipelineContractError, match="does not implement 'predict_proba'"):
        evaluator.evaluate(
            model_cfg=cast(TrainModelConfig, SimpleNamespace(task=SimpleNamespace(type="classification", subtype="binary"))),
            strict=False,
            best_threshold=0.5,
            train_dir=tmp_path,
        )


def test_evaluate_loads_artifacts_and_returns_evaluate_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run full orchestration and return expected ``EvaluateOutput`` payload."""
    module = _get_classification_module()

    class _Pipeline:
        """Minimal classifier test double exposing ``predict_proba``."""

        def predict_proba(self, X: pd.DataFrame) -> list[list[float]]:
            """Return deterministic probabilities sized to the input dataframe."""
            return [[0.4, 0.6] for _ in range(len(X))]

    pipeline_path = tmp_path / "pipeline.pkl"
    training_metadata = SimpleNamespace(artifacts=SimpleNamespace(pipeline_path=str(pipeline_path)))

    monkeypatch.setattr(
        module,
        "load_json",
        lambda _path: {"artifacts": {"pipeline_path": str(pipeline_path)}},
    )
    monkeypatch.setattr(
        module,
        "validate_training_metadata",
        lambda _raw: training_metadata,
    )
    monkeypatch.setattr(
        module,
        "load_model_or_pipeline",
        lambda _path, _kind: _Pipeline(),
    )
    monkeypatch.setattr(
        module,
        "get_snapshot_binding_from_training_metadata",
        lambda _meta: {"booking_context_features": "2026-03-07"},
    )

    snapshot_selection = {"booking_context_features": "2026-03-07"}
    monkeypatch.setattr(
        module,
        "resolve_feature_snapshots",
        lambda **_kwargs: snapshot_selection,
    )

    X = pd.DataFrame({"row_id": ["r1", "r2", "r3"], "feature": [0.1, 0.2, 0.3]})
    y = pd.Series([0, 1, 0], name="target")
    lineage = [SimpleNamespace(feature_set="booking_context_features")]
    monkeypatch.setattr(
        module,
        "load_features_and_target",
        lambda *_args, **_kwargs: (X, y, lineage, "entity_key"),
    )

    validate_snapshot_ids_calls: list[tuple[Any, Any]] = []

    def _validate_snapshot_ids(feature_lineage: Any, selected: Any) -> None:
        validate_snapshot_ids_calls.append((feature_lineage, selected))

    monkeypatch.setattr(
        module,
        "validate_snapshot_ids",
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
        module,
        "get_splits",
        lambda **_kwargs: (splits, {"summary": "ok"}),
    )

    expected_metrics = {"train": {"accuracy": 1.0}, "val": {"accuracy": 1.0}, "test": {"accuracy": 1.0}}
    expected_prediction_artifacts = SimpleNamespace(train=pd.DataFrame(), val=pd.DataFrame(), test=pd.DataFrame())
    evaluate_model_calls: list[dict[str, Any]] = []

    def _evaluate_model(*args: Any, **kwargs: Any) -> tuple[dict[str, dict[str, float]], Any]:
        evaluate_model_calls.append({"args": args, "kwargs": kwargs})
        return expected_metrics, expected_prediction_artifacts

    monkeypatch.setattr(
        module,
        "evaluate_model",
        _evaluate_model,
    )

    model_cfg = cast(
        TrainModelConfig,
        SimpleNamespace(
            feature_store=SimpleNamespace(path=str(tmp_path), feature_sets=["booking_context_features"]),
            split=SimpleNamespace(type="random", train_size=0.7),
            data_type="batch",
            task=SimpleNamespace(type="classification", subtype="binary"),
            target=SimpleNamespace(transform=SimpleNamespace(enabled=False, type=None, lambda_value=None)),
        ),
    )

    evaluator = _get_classification_evaluator()()
    output = evaluator.evaluate(
        model_cfg=model_cfg,
        strict=True,
        best_threshold=0.42,
        train_dir=tmp_path,
    )

    assert isinstance(output, EvaluateOutput)
    assert output.metrics == expected_metrics
    assert output.prediction_dfs is expected_prediction_artifacts
    assert output.lineage == lineage

    assert validate_snapshot_ids_calls == [(lineage, snapshot_selection)]
    assert len(evaluate_model_calls) == 1
    call_kwargs = evaluate_model_calls[0]["kwargs"]
    call_args = evaluate_model_calls[0]["args"]
    assert hasattr(call_kwargs["pipeline"], "predict_proba")
    assert call_kwargs["best_threshold"] == 0.42
    assert call_args[0] is model_cfg

    data_splits = call_kwargs["data_splits"]
    assert data_splits.train[0].equals(splits.X_train)
    assert data_splits.val[0].equals(splits.X_val)
    assert data_splits.test[0].equals(splits.X_test)
