"""Unit tests for interim data-preparation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import cast

import pandas as pd
import pytest
from ml.data.config.schemas.interim import Cleaning, DataSchema, Invariants
from ml.data.interim.data_preparation.prepare_data import (
    clean_data,
    enforce_schema,
    normalize_columns,
)
from ml.exceptions import DataError

pytestmark = pytest.mark.unit


@dataclass
class _SchemaStub:
    """Minimal schema stub exposing a ``model_dump`` mapping."""

    values: dict[str, str]

    def model_dump(self) -> dict[str, str]:
        """Return schema mapping used by ``enforce_schema``."""
        return self.values


@dataclass
class _Bound:
    """Boundary descriptor compatible with invariant operators."""

    op: str
    value: float


@dataclass
class _Rules:
    """Column rules container for min/max/allowed-values checks."""

    min: _Bound | None = None
    max: _Bound | None = None
    allowed_values: list[str] | None = None


def test_normalize_columns_applies_enabled_transformations_in_order() -> None:
    """Normalize mixed-case, spaced, and dashed column names into stable keys."""
    df = pd.DataFrame(columns=["  Arrival Date-Year  ", "Guest Name"])
    cleaning = cast(
        Cleaning,
        SimpleNamespace(
            lowercase_columns=True,
            strip_strings=True,
            replace_spaces_in_columns=True,
            replace_dashes_in_columns=True,
        ),
    )

    normalized = normalize_columns(df, cleaning)

    assert list(normalized.columns) == ["arrival_date_year", "guest_name"]


def test_enforce_schema_drops_extra_columns_and_casts_supported_types() -> None:
    """Drop unknown columns and cast integer, datetime, and category schema dtypes."""
    schema = cast(
        DataSchema,
        _SchemaStub(
            {
                "is_canceled": "int8",
                "reservation_status_date": "datetime64[ns]",
                "hotel": "category",
            }
        ),
    )
    df = pd.DataFrame(
        {
            "is_canceled": [0, 1],
            "reservation_status_date": ["2017-01-01", "2017-01-02"],
            "hotel": ["City Hotel", "Resort Hotel"],
            "extra_col": ["x", "y"],
        }
    )

    result = enforce_schema(df, schema=schema, drop_missing_ints=False)

    assert list(result.columns) == ["is_canceled", "reservation_status_date", "hotel"]
    assert str(result["is_canceled"].dtype) == "int8"
    assert str(result["reservation_status_date"].dtype).startswith("datetime64")
    assert str(result["hotel"].dtype) == "category"


def test_enforce_schema_converts_nullable_integer_to_float_when_not_dropping() -> None:
    """Preserve rows and widen nullable integer columns to float when configured."""
    schema = cast(DataSchema, _SchemaStub({"lead_time": "int16"}))
    df = pd.DataFrame({"lead_time": [1, None, 3]})

    result = enforce_schema(df, schema=schema, drop_missing_ints=False)

    assert len(result) == 3
    assert str(result["lead_time"].dtype) == "float64"
    assert pd.isna(result.iloc[1]["lead_time"])


def test_enforce_schema_drops_rows_with_missing_integer_values_when_enabled() -> None:
    """Drop only rows violating non-null integer requirements when enabled."""
    schema = cast(DataSchema, _SchemaStub({"lead_time": "int16", "hotel": "category"}))
    df = pd.DataFrame(
        {
            "lead_time": [1, None, 3],
            "hotel": ["City Hotel", "Resort Hotel", "City Hotel"],
        }
    )

    result = enforce_schema(df, schema=schema, drop_missing_ints=True)

    assert len(result) == 2
    assert list(result["lead_time"].astype(int)) == [1, 3]


def test_enforce_schema_raises_data_error_when_required_columns_are_missing() -> None:
    """Raise ``DataError`` when dataframe does not contain all schema columns."""
    schema = cast(DataSchema, _SchemaStub({"lead_time": "int16", "hotel": "category"}))
    df = pd.DataFrame({"lead_time": [1, 2]})

    with pytest.raises(DataError, match="Error enforcing data schema"):
        enforce_schema(df, schema=schema, drop_missing_ints=True)


def test_clean_data_applies_min_max_and_allowed_values_while_keeping_null_or_empty() -> None:
    """Filter rows by invariants and preserve null/empty values per function contract."""
    df = pd.DataFrame(
        {
            "lead_time": [-1, 5, 11, None, 3],
            "hotel": [" City Hotel ", " Resort Hotel ", " City Hotel ", "", " City Hotel "],
        }
    )
    invariants = cast(
        Invariants,
        SimpleNamespace(
            lead_time=_Rules(min=_Bound(op="gte", value=0), max=_Bound(op="lte", value=10)),
            hotel=_Rules(allowed_values=["City Hotel"]),
        ),
    )

    result = clean_data(df, invariants)

    assert list(result.index) == [3, 4]
    assert result.loc[4, "hotel"] == "City Hotel"
    assert pd.isna(result.loc[3, "lead_time"])
    assert result.loc[3, "hotel"] == ""


def test_clean_data_wraps_operator_failures_as_data_error() -> None:
    """Convert internal invariant-evaluation errors into stable ``DataError`` failures."""
    df = pd.DataFrame({"lead_time": [1, 2]})
    invariants = cast(
        Invariants,
        SimpleNamespace(
            lead_time=_Rules(min=_Bound(op="does_not_exist", value=0)),
        ),
    )

    with pytest.raises(DataError, match="Error cleaning data according to invariants"):
        clean_data(df, invariants)
