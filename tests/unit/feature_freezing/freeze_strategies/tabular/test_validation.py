"""Unit tests for tabular freeze validation helpers."""

from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest
from ml.exceptions import DataError
from ml.feature_freezing.freeze_strategies.tabular.validation import (
    validate_constraints,
    validate_data_types,
    validate_input_no_nulls,
    validate_max_cardinality,
)

pytestmark = pytest.mark.unit


def _config_stub(
    *,
    forbid_nulls: list[str] | None = None,
    max_cardinality: dict[str, int] | None = None,
    categorical: list[str] | None = None,
    numerical: list[str] | None = None,
    datetime: list[str] | None = None,
) -> Any:
    """Create minimal config-like object exposing fields consumed by validators."""
    cfg = SimpleNamespace(
        constraints=SimpleNamespace(
            forbid_nulls=forbid_nulls or [],
            max_cardinality=max_cardinality or {},
        ),
        feature_roles=SimpleNamespace(
            categorical=categorical or [],
            numerical=numerical or [],
            datetime=datetime or [],
        ),
    )
    return cast(Any, cfg)


def test_validate_input_no_nulls_raises_when_forbidden_column_contains_null() -> None:
    """Reject input when configured no-null column includes missing values."""
    X = pd.DataFrame({"hotel": ["A", None]})
    cfg = _config_stub(forbid_nulls=["hotel"])

    with pytest.raises(DataError, match="contains null values"):
        validate_input_no_nulls(X, cfg)


def test_validate_input_no_nulls_ignores_columns_not_in_forbidden_list() -> None:
    """Allow nulls in columns that are not listed as forbidden by constraints."""
    X = pd.DataFrame({"hotel": ["A", None]})
    cfg = _config_stub(forbid_nulls=["market_segment"])

    validate_input_no_nulls(X, cfg)


def test_validate_max_cardinality_raises_when_limit_exceeded() -> None:
    """Reject categorical feature values when uniqueness exceeds configured cap."""
    X = pd.DataFrame({"agent": ["a", "b", "c"]})
    cfg = _config_stub(categorical=["agent"], max_cardinality={"agent": 2})

    with pytest.raises(DataError, match="exceeds max cardinality"):
        validate_max_cardinality(X, cfg)


def test_validate_data_types_raises_for_invalid_categorical_dtype() -> None:
    """Reject categorical role columns whose normalized dtype is not allowed."""
    X = pd.DataFrame({"cat_col": pd.Series([1, 2, 3], dtype="int64")})
    cfg = _config_stub(categorical=["cat_col"])

    with pytest.raises(DataError, match="Categorical feature cat_col has invalid dtype"):
        validate_data_types(X, cfg)


def test_validate_data_types_raises_for_invalid_numerical_dtype() -> None:
    """Reject numerical role columns with non-numeric normalized dtypes."""
    X = pd.DataFrame({"num_col": pd.Series(["x", "y"], dtype="object")})
    cfg = _config_stub(numerical=["num_col"])

    with pytest.raises(DataError, match="Numerical feature num_col has invalid dtype"):
        validate_data_types(X, cfg)


def test_validate_data_types_accepts_valid_role_dtypes() -> None:
    """Pass when all role-assigned columns satisfy configured dtype allowlists."""
    X = pd.DataFrame(
        {
            "cat_col": pd.Series(["a", "b"], dtype="object"),
            "num_col": pd.Series([1.5, 2.5], dtype="float64"),
            "dt_col": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        }
    )
    cfg = _config_stub(
        categorical=["cat_col"],
        numerical=["num_col"],
        datetime=["dt_col"],
    )

    validate_data_types(X, cfg)


def test_validate_constraints_runs_both_null_and_cardinality_checks() -> None:
    """Run combined constraints and fail when any delegated validation fails."""
    X = pd.DataFrame({"agent": ["a", "b", "c"]})
    cfg = _config_stub(
        forbid_nulls=["agent"],
        categorical=["agent"],
        max_cardinality={"agent": 2},
    )

    with pytest.raises(DataError, match="exceeds max cardinality"):
        validate_constraints(X, cfg)
