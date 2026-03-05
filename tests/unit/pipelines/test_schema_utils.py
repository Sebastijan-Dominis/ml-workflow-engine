"""Unit tests for schema-based pipeline feature grouping utilities."""

from types import SimpleNamespace
from typing import cast

import pandas as pd
import pytest
from ml.config.schemas.model_cfg import SearchModelConfig
from ml.pipelines.schema_utils import get_categorical_features, get_pipeline_features

pytestmark = pytest.mark.unit


def test_get_categorical_features_keeps_supported_dtype_features_in_order() -> None:
    """Select object/string/category rows only and preserve schema row order."""
    schema = pd.DataFrame(
        {
            "feature": ["country", "adults", "market_segment", "price", "reserved_room_type"],
            "dtype": ["object", "int64", "string", "float64", "category"],
        }
    )

    result = get_categorical_features(schema)

    assert result == ["country", "market_segment", "reserved_room_type"]


def test_get_pipeline_features_excludes_segmentation_columns_when_not_modeled() -> None:
    """Drop segmentation filter columns from model input when segmentation is enabled-only-for-filtering."""
    input_schema = pd.DataFrame(
        {
            "feature": ["hotel", "market_segment", "lead_time", "adr"],
            "dtype": ["object", "string", "int64", "float64"],
        }
    )
    derived_schema = pd.DataFrame(
        {
            "feature": ["adr_per_person", "total_stay"],
            "dtype": ["float64", "int64"],
        }
    )
    model_cfg = cast(
        SearchModelConfig,
        SimpleNamespace(
            segmentation=SimpleNamespace(
                enabled=True,
                include_in_model=False,
                filters=[SimpleNamespace(column="hotel"), SimpleNamespace(column="market_segment")],
            )
        ),
    )

    result = get_pipeline_features(
        model_cfg,
        input_schema=input_schema,
        derived_schema=derived_schema,
    )

    assert result.input_features == ["lead_time", "adr"]
    assert result.derived_features == ["adr_per_person", "total_stay"]
    assert result.categorical_features == []
    assert result.selected_features == ["lead_time", "adr", "adr_per_person", "total_stay"]


def test_get_pipeline_features_keeps_segmentation_columns_when_included() -> None:
    """Keep all input columns when segmentation filters are also part of the model feature set."""
    input_schema = pd.DataFrame(
        {
            "feature": ["hotel", "lead_time", "distribution_channel"],
            "dtype": ["category", "int64", "object"],
        }
    )
    derived_schema = pd.DataFrame({"feature": ["total_stay"], "dtype": ["int64"]})
    model_cfg = cast(
        SearchModelConfig,
        SimpleNamespace(
            segmentation=SimpleNamespace(
                enabled=True,
                include_in_model=True,
                filters=[SimpleNamespace(column="hotel")],
            )
        ),
    )

    result = get_pipeline_features(
        model_cfg,
        input_schema=input_schema,
        derived_schema=derived_schema,
    )

    assert result.input_features == ["hotel", "lead_time", "distribution_channel"]
    assert result.categorical_features == ["hotel", "distribution_channel"]
    assert result.selected_features == ["hotel", "lead_time", "distribution_channel", "total_stay"]
