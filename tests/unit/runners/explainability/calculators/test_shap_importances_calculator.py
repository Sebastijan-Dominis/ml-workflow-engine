"""Unit tests for tree-model SHAP-importance calculator helper."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from ml.exceptions import ConfigError, DataError, ExplainabilityError
from ml.runners.explainability.explainers.tree_model.utils.calculators.shap_importances import (
    get_shap_importances,
)

pytestmark = pytest.mark.unit


class _AdapterStub:
    """Minimal adapter stub for controlled SHAP computation behavior."""

    def __init__(self, *, shap_values: Any = None, error: Exception | None = None) -> None:
        self.shap_values = shap_values
        self.error = error
        self.calls: list[pd.DataFrame] = []

    def compute_shap_values(self, X: pd.DataFrame) -> Any:
        """Capture sampled frame and return configured SHAP values or raise."""
        self.calls.append(X)
        if self.error is not None:
            raise self.error
        return self.shap_values


def _cfg(*, enabled: bool, approximate: str) -> Any:
    """Create minimal config-like object matching nested shap settings access."""
    return SimpleNamespace(
        explainability=SimpleNamespace(
            methods=SimpleNamespace(shap=SimpleNamespace(enabled=enabled, approximate=approximate))
        )
    )


def test_get_shap_importances_returns_none_when_disabled() -> None:
    """Skip SHAP computation when disabled and return None."""
    adapter = _AdapterStub(shap_values=np.zeros((1, 1)))

    result = get_shap_importances(
        feature_names=np.array(["f1"], dtype=np.str_),
        model_configs=_cfg(enabled=False, approximate="tree"),  # type: ignore[arg-type]
        top_k=1,
        X_test_transformed=pd.DataFrame({"f1": [1.0]}),
        adapter=adapter,  # type: ignore[arg-type]
    )

    assert result is None
    assert adapter.calls == []


def test_get_shap_importances_rejects_unsupported_method() -> None:
    """Raise ConfigError when configured SHAP approximation method is unsupported."""
    with pytest.raises(ConfigError, match="Unsupported SHAP method: kernel"):
        get_shap_importances(
            feature_names=np.array(["f1"], dtype=np.str_),
            model_configs=_cfg(enabled=True, approximate="kernel"),  # type: ignore[arg-type]
            top_k=1,
            X_test_transformed=pd.DataFrame({"f1": [1.0]}),
            adapter=_AdapterStub(shap_values=np.zeros((1, 1))),  # type: ignore[arg-type]
        )


def test_get_shap_importances_wraps_adapter_failures_as_explainability_error() -> None:
    """Wrap adapter SHAP computation failures with ExplainabilityError."""
    adapter = _AdapterStub(error=RuntimeError("shap failed"))

    with pytest.raises(ExplainabilityError, match="Error calculating SHAP values") as exc_info:
        get_shap_importances(
            feature_names=np.array(["f1"], dtype=np.str_),
            model_configs=_cfg(enabled=True, approximate="tree"),  # type: ignore[arg-type]
            top_k=1,
            X_test_transformed=pd.DataFrame({"f1": [1.0], "f2": [2.0]}),
            adapter=adapter,  # type: ignore[arg-type]
        )

    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_get_shap_importances_computes_sorted_top_k_from_array_output() -> None:
    """Aggregate absolute SHAP array output to mean importances and sort top-k."""
    shap_values = np.array(
        [
            [1.0, -2.0, 0.5],
            [-3.0, 1.0, -0.5],
            [2.0, -4.0, 1.0],
        ],
        dtype=np.float64,
    )
    adapter = _AdapterStub(shap_values=shap_values)

    result = get_shap_importances(
        feature_names=np.array(["f1", "f2", "f3"], dtype=np.str_),
        model_configs=_cfg(enabled=True, approximate="tree"),  # type: ignore[arg-type]
        top_k=2,
        X_test_transformed=pd.DataFrame(
            {"f1": [10.0, 11.0, 12.0], "f2": [1.0, 2.0, 3.0], "f3": [5.0, 6.0, 7.0]}
        ),
        adapter=adapter,  # type: ignore[arg-type]
    )

    assert result is not None
    assert result.to_dict(orient="records") == [
        {"feature": "f2", "mean_abs_shap": 2.3333333333333335},
        {"feature": "f1", "mean_abs_shap": 2.0},
    ]


def test_get_shap_importances_accepts_multiclass_list_output() -> None:
    """Reduce list-based SHAP outputs by averaging abs values across classes first."""
    shap_values = [
        np.array([[1.0, -2.0], [-1.0, 2.0]], dtype=np.float64),
        np.array([[3.0, -4.0], [-3.0, 4.0]], dtype=np.float64),
    ]
    adapter = _AdapterStub(shap_values=shap_values)

    result = get_shap_importances(
        feature_names=np.array(["f1", "f2"], dtype=np.str_),
        model_configs=_cfg(enabled=True, approximate="tree"),  # type: ignore[arg-type]
        top_k=2,
        X_test_transformed=pd.DataFrame({"f1": [0.0, 1.0], "f2": [2.0, 3.0]}),
        adapter=adapter,  # type: ignore[arg-type]
    )

    assert result is not None
    assert result.to_dict(orient="records") == [
        {"feature": "f2", "mean_abs_shap": 3.0},
        {"feature": "f1", "mean_abs_shap": 2.0},
    ]


def test_get_shap_importances_rejects_non_dataframe_input() -> None:
    """Raise DataError when transformed test data lacks DataFrame interface."""
    with pytest.raises(DataError, match="Transformed test data is not a pandas DataFrame"):
        get_shap_importances(
            feature_names=np.array(["f1"], dtype=np.str_),
            model_configs=_cfg(enabled=True, approximate="tree"),  # type: ignore[arg-type]
            top_k=1,
            X_test_transformed=np.array([[1.0], [2.0]]),  # type: ignore[arg-type]
            adapter=_AdapterStub(shap_values=np.array([[0.1], [0.2]])),  # type: ignore[arg-type]
        )
