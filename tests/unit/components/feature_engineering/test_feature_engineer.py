"""Unit tests for feature-engineering orchestration and output contracts."""

import pandas as pd
import pytest
from ml.components.feature_engineering.base import FeatureEngineer
from ml.exceptions import DataError

pytestmark = pytest.mark.unit


class _AddOneFeature:
    """Test operator that appends a deterministic derived feature."""

    output_features = ["f1"]

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["f1"] = X["base"] + 1
        return X


class _AddSecondFeature:
    """Test operator that depends on prior derived feature output."""

    output_features = ["f2"]

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["f2"] = X["f1"] * 2
        return X


class _BrokenOperator:
    """Test operator that violates output contract by omitting expected feature."""

    output_features = ["missing_feature"]

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.copy()


def test_feature_engineer_applies_operators_in_schema_order() -> None:
    """Execute operators in derived-schema order and propagate intermediate outputs."""
    derived_schema = pd.DataFrame(
        {
            "source_operator": [
                "first",
                "first",  # duplicate source_operator should execute only once via unique()
                "second",
            ]
        }
    )
    engineer = FeatureEngineer(
        derived_schema=derived_schema,
        operators={"first": _AddOneFeature(), "second": _AddSecondFeature()},
    )
    df = pd.DataFrame({"base": [1, 2]})

    transformed = engineer.transform(df)

    assert transformed["f1"].tolist() == [2, 3]
    assert transformed["f2"].tolist() == [4, 6]
    assert "f1" not in df.columns
    assert "f2" not in df.columns


def test_feature_engineer_raises_when_operator_does_not_emit_expected_feature() -> None:
    """Fail fast when an operator violates declared output feature contract."""
    derived_schema = pd.DataFrame({"source_operator": ["broken"]})
    engineer = FeatureEngineer(
        derived_schema=derived_schema,
        operators={"broken": _BrokenOperator()},
    )
    df = pd.DataFrame({"base": [1, 2]})

    with pytest.raises(DataError, match="did not produce expected feature"):
        engineer.transform(df)
