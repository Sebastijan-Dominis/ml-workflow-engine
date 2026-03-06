"""Unit tests for CatBoost tree-model adapter implementation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from ml.exceptions import ExplainabilityError
from ml.runners.explainability.explainers.tree_model.adapters.catboost import (
    CatBoostAdapter,
)

pytestmark = pytest.mark.unit


class _ModelStub:
    """CatBoost-like model stub capturing adapter method calls."""

    def __init__(
        self,
        *,
        cat_indices: list[int] | None = None,
        shap_values: Any = None,
        feature_importances: Any = None,
        error: Exception | None = None,
    ) -> None:
        self._cat_indices = cat_indices or []
        self._shap_values = shap_values
        self._feature_importances = feature_importances
        self._error = error
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get_cat_feature_indices(self) -> list[int]:
        """Return configured categorical feature indices."""
        self.calls.append(("get_cat_feature_indices", {}))
        return self._cat_indices

    def get_feature_importance(self, **kwargs: Any) -> Any:
        """Capture invocation and return SHAP/importances payload or raise."""
        self.calls.append(("get_feature_importance", kwargs))
        if self._error is not None:
            raise self._error
        if kwargs.get("type") == "ShapValues":
            return self._shap_values
        return self._feature_importances


def test_compute_shap_values_builds_pool_and_drops_expected_value_column(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Construct Pool with cat features and strip last SHAP expected-value column."""
    pool_calls: list[dict[str, Any]] = []

    class _PoolStub:
        def __init__(self, *, data: pd.DataFrame, cat_features: list[int]) -> None:
            pool_calls.append({"data": data, "cat_features": cat_features})

    shap_matrix = np.array([[1.0, 2.0, 9.0], [3.0, 4.0, 9.0]], dtype=np.float64)
    model = _ModelStub(cat_indices=[0], shap_values=shap_matrix)
    adapter = CatBoostAdapter(model)
    X = pd.DataFrame({"cat_col": ["a", "b"], "num_col": [1.0, 2.0]})

    monkeypatch.setattr(
        "ml.runners.explainability.explainers.tree_model.adapters.catboost.Pool",
        _PoolStub,
    )

    result = adapter.compute_shap_values(X)

    assert pool_calls == [{"data": X, "cat_features": [0]}]
    assert model.calls[0][0] == "get_cat_feature_indices"
    assert model.calls[1][0] == "get_feature_importance"
    assert model.calls[1][1]["type"] == "ShapValues"
    assert "data" in model.calls[1][1]
    assert result.tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_compute_shap_values_wraps_pool_or_model_failures() -> None:
    """Wrap SHAP computation failures as ExplainabilityError with preserved cause."""
    model = _ModelStub(error=RuntimeError("shap failed"))
    adapter = CatBoostAdapter(model)
    X = pd.DataFrame({"f": [1.0]})

    with pytest.raises(ExplainabilityError, match="Error computing SHAP values for CatBoost model") as exc_info:
        adapter.compute_shap_values(X)

    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_compute_feature_importances_forwards_importance_type() -> None:
    """Forward requested importance type and return numpy-converted values."""
    model = _ModelStub(feature_importances=[0.1, 0.9, 0.3])
    adapter = CatBoostAdapter(model)

    result = adapter.compute_feature_importances("PredictionValuesChange")

    assert isinstance(result, np.ndarray)
    assert result.tolist() == [0.1, 0.9, 0.3]
    assert model.calls == [
        ("get_feature_importance", {"type": "PredictionValuesChange"}),
    ]


def test_compute_feature_importances_wraps_failures_with_context() -> None:
    """Wrap feature-importance failures as ExplainabilityError with cause preserved."""
    model = _ModelStub(error=ValueError("invalid importance type"))
    adapter = CatBoostAdapter(model)

    with pytest.raises(
        ExplainabilityError,
        match="Error computing CatBoost feature importances with type 'FeatureImportance'",
    ) as exc_info:
        adapter.compute_feature_importances("FeatureImportance")

    assert isinstance(exc_info.value.__cause__, ValueError)
