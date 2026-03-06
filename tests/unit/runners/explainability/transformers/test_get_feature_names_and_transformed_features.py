"""Unit tests for transformed feature-name extraction helper."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from ml.exceptions import DataError, PipelineContractError
from ml.runners.explainability.explainers.tree_model.utils.transformers.get_feature_names_and_transformed_features import (
    get_feature_names_and_transformed_features,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

pytestmark = pytest.mark.unit


def test_get_feature_names_and_transformed_features_returns_dataframe_columns() -> None:
    """Return transformed dataframe and its column names when transform preserves columns."""

    def _add_feature(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["lead_time_x2"] = out["lead_time"] * 2
        return out

    pipeline = Pipeline(
        steps=[
            ("prep", FunctionTransformer(_add_feature, validate=False)),
            ("model", object()),
        ]
    )
    X = pd.DataFrame({"lead_time": [5, 7], "adr": [100.0, 120.0]})

    feature_names, transformed = get_feature_names_and_transformed_features(pipeline, X)

    assert feature_names.tolist() == ["lead_time", "adr", "lead_time_x2"]
    assert transformed.equals(
        pd.DataFrame(
            {"lead_time": [5, 7], "adr": [100.0, 120.0], "lead_time_x2": [10, 14]}
        )
    )


def test_get_feature_names_and_transformed_features_wraps_transform_errors(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Wrap transformer failures as PipelineContractError with actionable message."""

    def _raise(df: pd.DataFrame) -> pd.DataFrame:
        _ = df
        raise ValueError("bad transform")

    pipeline = Pipeline(
        steps=[
            ("prep", FunctionTransformer(_raise, validate=False)),
            ("model", object()),
        ]
    )
    X = pd.DataFrame({"lead_time": [1]})

    with caplog.at_level(
        "ERROR",
        logger=(
            "ml.runners.explainability.explainers.tree_model.utils.transformers."
            "get_feature_names_and_transformed_features"
        ),
    ), pytest.raises(
        PipelineContractError,
        match="Error transforming data using the pipeline",
    ) as exc_info:
        get_feature_names_and_transformed_features(pipeline, X)

    assert isinstance(exc_info.value.__cause__, ValueError)
    assert "Error transforming data using the pipeline" in caplog.text


def test_get_feature_names_and_transformed_features_rejects_transform_without_columns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Raise DataError when transformed output does not expose dataframe columns."""
    pipeline = Pipeline(
        steps=[
            (
                "prep",
                FunctionTransformer(lambda df: np.asarray(df), validate=False),
            ),
            ("model", object()),
        ]
    )
    X = pd.DataFrame({"lead_time": [3, 4]})

    with caplog.at_level(
        "ERROR",
        logger=(
            "ml.runners.explainability.explainers.tree_model.utils.transformers."
            "get_feature_names_and_transformed_features"
        ),
    ), pytest.raises(
        DataError,
        match="Transformed data has no column names",
    ):
        get_feature_names_and_transformed_features(pipeline, X)

    assert "Transformed data has no column names" in caplog.text
