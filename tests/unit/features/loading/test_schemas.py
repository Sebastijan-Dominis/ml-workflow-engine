"""Unit tests for feature schema loading and aggregation helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pandas as pd
import pytest
from ml.config.schemas.model_cfg import SearchModelConfig
from ml.exceptions import DataError
from ml.features.loading.schemas import aggregate_schema_dfs, load_feature_set_schemas, load_schemas

pytestmark = pytest.mark.unit


def test_load_feature_set_schemas_raises_when_input_schema_is_missing(tmp_path: Path) -> None:
    """Raise ``DataError`` when the required input schema CSV file is absent."""
    with pytest.raises(DataError, match="Input schema file not found"):
        load_feature_set_schemas(tmp_path)


def test_load_feature_set_schemas_returns_empty_derived_when_derived_schema_missing(tmp_path: Path) -> None:
    """Load input schema and return an empty derived schema when derived CSV is absent."""
    input_schema_path = tmp_path / "input_schema.csv"
    pd.DataFrame({"feature": ["lead_time"], "dtype": ["int64"]}).to_csv(input_schema_path, index=False)

    input_schema, derived_schema = load_feature_set_schemas(tmp_path)

    assert input_schema.shape == (1, 2)
    assert derived_schema.empty


def test_aggregate_schema_dfs_returns_empty_dataframe_for_empty_input() -> None:
    """Return an empty dataframe when no schema inputs are provided."""
    aggregated = aggregate_schema_dfs([])

    assert aggregated.empty


def test_aggregate_schema_dfs_raises_when_feature_column_is_missing() -> None:
    """Raise ``DataError`` if any schema dataframe lacks the required ``feature`` column."""
    bad_schema = pd.DataFrame({"name": ["lead_time"]})

    with pytest.raises(DataError, match="Schema must contain a 'feature' column"):
        aggregate_schema_dfs([bad_schema])


def test_aggregate_schema_dfs_preserves_first_occurrence_and_order() -> None:
    """Deduplicate by feature name while preserving first-seen row order."""
    schema_a = pd.DataFrame(
        {
            "feature": ["lead_time", "hotel"],
            "dtype": ["int64", "category"],
        }
    )
    schema_b = pd.DataFrame(
        {
            "feature": ["hotel", "adr", "lead_time"],
            "dtype": ["string", "float64", "int32"],
        }
    )

    aggregated = aggregate_schema_dfs([schema_a, schema_b])

    assert aggregated["feature"].tolist() == ["lead_time", "hotel", "adr"]
    assert aggregated.loc[0, "dtype"] == "int64"
    assert aggregated.loc[1, "dtype"] == "category"


def test_load_schemas_raises_when_no_feature_sets_defined() -> None:
    """Raise ``DataError`` when model configuration has no feature sets."""
    model_cfg = cast(
        SearchModelConfig,
        SimpleNamespace(feature_store=SimpleNamespace(path="feature_store", feature_sets=[])),
    )

    with pytest.raises(DataError, match="No feature sets defined"):
        load_schemas(model_cfg)


def test_load_schemas_aggregates_input_and_derived_schemas_across_feature_sets(tmp_path: Path) -> None:
    """Aggregate all configured feature-set schemas and deduplicate by feature name."""
    fs_root = tmp_path / "feature_store"

    fs_a = fs_root / "booking_context_features" / "v1"
    fs_b = fs_root / "pricing_party_features" / "v2"
    fs_a.mkdir(parents=True)
    fs_b.mkdir(parents=True)

    pd.DataFrame(
        {
            "feature": ["lead_time", "hotel"],
            "dtype": ["int64", "category"],
        }
    ).to_csv(fs_a / "input_schema.csv", index=False)
    pd.DataFrame(
        {
            "feature": ["is_long_stay"],
            "source_operator": ["length_bucket"],
        }
    ).to_csv(fs_a / "derived_schema.csv", index=False)

    pd.DataFrame(
        {
            "feature": ["hotel", "adr"],
            "dtype": ["string", "float64"],
        }
    ).to_csv(fs_b / "input_schema.csv", index=False)

    model_cfg = cast(
        SearchModelConfig,
        SimpleNamespace(
            feature_store=SimpleNamespace(
                path=str(fs_root),
                feature_sets=[
                    SimpleNamespace(name="booking_context_features", version="v1"),
                    SimpleNamespace(name="pricing_party_features", version="v2"),
                ],
            )
        ),
    )

    input_schema, derived_schema = load_schemas(model_cfg)

    assert input_schema["feature"].tolist() == ["lead_time", "hotel", "adr"]
    assert derived_schema["feature"].tolist() == ["is_long_stay"]
