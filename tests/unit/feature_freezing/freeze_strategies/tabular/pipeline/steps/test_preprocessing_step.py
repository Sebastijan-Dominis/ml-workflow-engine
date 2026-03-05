"""Unit tests for tabular preprocessing pipeline step orchestration."""

import importlib
import sys
import types
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest

pytestmark = pytest.mark.unit


@pytest.fixture()
def preprocessing_module(monkeypatch: pytest.MonkeyPatch):
    """Import preprocessing step module with heavy dependencies stubbed out."""
    sys.modules.pop(
        "ml.feature_freezing.freeze_strategies.tabular.pipeline.steps.preprocessing",
        None,
    )

    features_module = cast(
        Any,
        types.ModuleType("ml.feature_freezing.freeze_strategies.tabular.features"),
    )
    validation_module = cast(
        Any,
        types.ModuleType("ml.feature_freezing.freeze_strategies.tabular.validation"),
    )
    context_module = cast(
        Any,
        types.ModuleType("ml.feature_freezing.freeze_strategies.tabular.pipeline.context"),
    )

    features_module.prepare_features = lambda data, config: data.copy()
    features_module.apply_operators = lambda X, operator_names, required_features: X.assign(op_applied=True)
    validation_module.validate_data_types = lambda X, config: None
    validation_module.validate_constraints = lambda X, config: None
    context_module.FreezeContext = object

    monkeypatch.setitem(
        sys.modules,
        "ml.feature_freezing.freeze_strategies.tabular.features",
        features_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "ml.feature_freezing.freeze_strategies.tabular.validation",
        validation_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "ml.feature_freezing.freeze_strategies.tabular.pipeline.context",
        context_module,
    )

    return importlib.import_module(
        "ml.feature_freezing.freeze_strategies.tabular.pipeline.steps.preprocessing"
    )


def test_preprocessing_step_sets_features_without_operators(preprocessing_module) -> None:
    """Populate context features from prepared data when no operators configured."""
    step = preprocessing_module.PreprocessingStep()
    ctx = SimpleNamespace(
        require_data=pd.DataFrame({"row_id": [1], "x": [2]}),
        config=SimpleNamespace(operators=None),
        features=None,
    )

    out = step.run(ctx)

    assert out is ctx
    assert isinstance(ctx.features, pd.DataFrame)
    assert "op_applied" not in ctx.features.columns


def test_preprocessing_step_applies_materialized_operators(preprocessing_module) -> None:
    """Apply operators only when operator mode is explicitly materialized."""
    step = preprocessing_module.PreprocessingStep()
    ctx = SimpleNamespace(
        require_data=pd.DataFrame({"row_id": [1], "x": [2]}),
        config=SimpleNamespace(
            operators=SimpleNamespace(mode="materialized", names=["op"], required_features={"op": ["x"]})
        ),
        features=None,
    )

    step.run(ctx)

    assert "op_applied" in ctx.features.columns


def test_preprocessing_step_skips_operator_application_for_non_materialized_mode(
    preprocessing_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Skip operator application when mode is not materialized."""
    step = preprocessing_module.PreprocessingStep()
    ctx = SimpleNamespace(
        require_data=pd.DataFrame({"row_id": [1], "x": [2]}),
        config=SimpleNamespace(
            operators=SimpleNamespace(mode="lazy", names=["op"], required_features={"op": ["x"]})
        ),
        features=None,
    )

    called = {"apply": False}

    def _apply(*args: object, **kwargs: object) -> pd.DataFrame:
        called["apply"] = True
        return pd.DataFrame({"x": [1]})

    monkeypatch.setattr(preprocessing_module, "apply_operators", _apply)

    step.run(ctx)

    assert called["apply"] is False
