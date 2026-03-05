"""Unit tests for categorical feature extraction utilities."""

from types import SimpleNamespace
from typing import cast

import pandas as pd
import pytest
from ml.config.schemas.model_cfg import SearchModelConfig
from ml.features.extraction.cat_features import get_cat_features

pytestmark = pytest.mark.unit


def test_get_cat_features_combines_input_and_derived_categorical_columns() -> None:
    """Return categorical columns from both input and derived schemas in order."""
    model_cfg = cast(
        SearchModelConfig,
        SimpleNamespace(
            segmentation=SimpleNamespace(
                enabled=False,
                include_in_model=False,
                filters=[],
            )
        ),
    )
    input_schema = pd.DataFrame(
        {
            "feature": ["hotel", "lead_time", "country"],
            "dtype": ["category", "int64", "object"],
        }
    )
    derived_schema = pd.DataFrame(
        {
            "feature": ["arrival_season", "total_stay"],
            "dtype": ["string", "int64"],
        }
    )

    result = get_cat_features(model_cfg, input_schema, derived_schema)

    assert result == ["hotel", "country", "arrival_season"]


def test_get_cat_features_excludes_segmentation_columns_when_not_in_model() -> None:
    """Drop segmentation filter columns from input categoricals when excluded from modeling."""
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
    input_schema = pd.DataFrame(
        {
            "feature": ["hotel", "market_segment", "country", "adr"],
            "dtype": ["category", "string", "object", "float64"],
        }
    )
    derived_schema = pd.DataFrame({"feature": ["arrival_season"], "dtype": ["category"]})

    result = get_cat_features(model_cfg, input_schema, derived_schema)

    assert result == ["country", "arrival_season"]


def test_get_cat_features_keeps_segmentation_columns_when_included_in_model() -> None:
    """Preserve segmentation filter columns when config includes them as model inputs."""
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
    input_schema = pd.DataFrame(
        {
            "feature": ["hotel", "country", "lead_time"],
            "dtype": ["category", "object", "int64"],
        }
    )
    derived_schema = pd.DataFrame({"feature": ["total_stay"], "dtype": ["int64"]})

    result = get_cat_features(model_cfg, input_schema, derived_schema)

    assert result == ["hotel", "country"]
