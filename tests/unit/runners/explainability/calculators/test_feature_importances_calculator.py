"""Unit tests for tree-model feature-importance calculator helper."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from ml.exceptions import ExplainabilityError, PipelineContractError
from ml.runners.explainability.explainers.tree_model.utils.calculators.feature_importances import (
    get_feature_importances,
)
from sklearn.pipeline import Pipeline

pytestmark = pytest.mark.unit


class _AdapterStub:
    """Minimal adapter stub with configurable importance behavior."""

    def __init__(self, importances: Any = None, error: Exception | None = None) -> None:
        self.importances = importances
        self.error = error
        self.calls: list[Any] = []

    def compute_feature_importances(self, *, importance_type: Any) -> Any:
        """Record the request and return configured importances or raise error."""
        self.calls.append(importance_type)
        if self.error is not None:
            raise self.error
        return self.importances


def _cfg(*, enabled: bool, importance_type: str | None) -> Any:
    """Create minimal config stub matching nested field access pattern."""
    return SimpleNamespace(
        explainability=SimpleNamespace(
            methods=SimpleNamespace(
                feature_importances=SimpleNamespace(enabled=enabled, type=importance_type)
            )
        )
    )


def test_get_feature_importances_returns_top_k_sorted_descending() -> None:
    """Compute top-k dataframe sorted by descending importance values."""
    feature_names = np.array(["adr", "lead_time", "total_stay"], dtype=np.str_)
    adapter = _AdapterStub(importances=np.array([0.2, 0.9, 0.5], dtype=np.float64))
    pipeline = Pipeline(steps=[("prep", object()), ("model", object())])

    result = get_feature_importances(
        feature_names=feature_names,
        adapter=adapter,  # type: ignore[arg-type]
        pipeline=pipeline,
        model_cfg=_cfg(enabled=True, importance_type="FeatureImportance"),  # type: ignore[arg-type]
        top_k=2,
    )

    assert adapter.calls == ["FeatureImportance"]
    assert result is not None
    assert result.to_dict(orient="records") == [
        {"feature": "lead_time", "importance": 0.9},
        {"feature": "total_stay", "importance": 0.5},
    ]


def test_get_feature_importances_returns_none_when_feature_importances_disabled() -> None:
    """Skip computation and return None when method is disabled in config."""
    adapter = _AdapterStub(importances=np.array([1.0]))
    pipeline = Pipeline(steps=[("prep", object()), ("model", object())])

    result = get_feature_importances(
        feature_names=np.array(["f1"], dtype=np.str_),
        adapter=adapter,  # type: ignore[arg-type]
        pipeline=pipeline,
        model_cfg=_cfg(enabled=False, importance_type="FeatureImportance"),  # type: ignore[arg-type]
        top_k=1,
    )

    assert result is None
    assert adapter.calls == []


def test_get_feature_importances_wraps_attribute_error_as_pipeline_contract_error() -> None:
    """Map adapter attribute errors to PipelineContractError with model type context."""
    adapter = _AdapterStub(error=AttributeError("missing get_feature_importance"))
    pipeline = Pipeline(steps=[("prep", object()), ("model", object())])

    with pytest.raises(PipelineContractError, match="does not have 'get_feature_importance'"):
        get_feature_importances(
            feature_names=np.array(["f1"], dtype=np.str_),
            adapter=adapter,  # type: ignore[arg-type]
            pipeline=pipeline,
            model_cfg=_cfg(enabled=True, importance_type="FeatureImportance"),  # type: ignore[arg-type]
            top_k=1,
        )


def test_get_feature_importances_wraps_other_errors_as_explainability_error() -> None:
    """Map non-attribute adapter failures to ExplainabilityError with method context."""
    adapter = _AdapterStub(error=RuntimeError("adapter failed"))
    pipeline = Pipeline(steps=[("prep", object()), ("model", object())])

    with pytest.raises(ExplainabilityError, match="Error retrieving feature importances") as exc_info:
        get_feature_importances(
            feature_names=np.array(["f1"], dtype=np.str_),
            adapter=adapter,  # type: ignore[arg-type]
            pipeline=pipeline,
            model_cfg=_cfg(enabled=True, importance_type="PredictionValuesChange"),  # type: ignore[arg-type]
            top_k=1,
        )

    assert isinstance(exc_info.value.__cause__, RuntimeError)
