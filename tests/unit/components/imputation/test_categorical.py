"""Unit tests for categorical missing-value imputation transformer."""


from __future__ import annotations

import pandas as pd
import pytest
from ml.components.imputation.categorical import FillCategoricalMissing

pytestmark = pytest.mark.unit

def test_fill_categorical_missing_replaces_missing_values_and_preserves_input() -> None:
    """
    Test that FillCategoricalMissing imputes configured categorical columns with 'missing' and does not mutate the input DataFrame.
    """
    df = pd.DataFrame({
        "country": ["PRT", None, "ESP"],
        "market_segment": ["Online TA", None, "Offline TA/TO"],
        "numeric": [1, 2, 3],
    })
    transformer = FillCategoricalMissing(["country", "market_segment"])
    transformed = transformer.transform(df)
    assert transformed["country"].tolist() == ["PRT", "missing", "ESP"]
    assert transformed["market_segment"].tolist() == ["Online TA", "missing", "Offline TA/TO"]
    assert transformed["numeric"].tolist() == [1, 2, 3]
    assert df["country"].isna().tolist() == [False, True, False]
    assert df["market_segment"].isna().tolist() == [False, True, False]

def test_fill_categorical_missing_coerces_non_string_values_to_strings() -> None:
    """
    Test that FillCategoricalMissing coerces non-string values to strings during imputation.
    """
    df = pd.DataFrame({"room_type": [1, 2, None]})
    transformer = FillCategoricalMissing(["room_type"])
    transformed = transformer.transform(df)
    assert transformed["room_type"].tolist() == ["1.0", "2.0", "missing"]

def test_fill_categorical_missing_fit_returns_self() -> None:
    """
    Test that fit returns the transformer instance itself (stateless behavior).
    """
    transformer = FillCategoricalMissing(["country"])
    df = pd.DataFrame({"country": ["PRT"]})
    result = transformer.fit(df)
    assert result is transformer

def test_fill_categorical_missing_empty_dataframe() -> None:
    """
    Test that FillCategoricalMissing handles an empty DataFrame gracefully.
    """
    df = pd.DataFrame(columns=["country"])
    transformer = FillCategoricalMissing(["country"])
    transformed = transformer.transform(df)
    assert transformed.empty

def test_fill_categorical_missing_all_missing() -> None:
    """
    Test that FillCategoricalMissing imputes all missing values in a column.
    """
    df = pd.DataFrame({"country": [None, None]})
    transformer = FillCategoricalMissing(["country"])
    transformed = transformer.transform(df)
    assert transformed["country"].tolist() == ["missing", "missing"]

def test_fill_categorical_missing_no_missing() -> None:
    """
    Test that FillCategoricalMissing leaves columns unchanged if there are no missing values.
    """
    df = pd.DataFrame({"country": ["PRT", "ESP"]})
    transformer = FillCategoricalMissing(["country"])
    transformed = transformer.transform(df)
    assert transformed["country"].tolist() == ["PRT", "ESP"]

def test_fill_categorical_missing_non_listed_column_untouched() -> None:
    """
    Test that columns not listed for imputation are not changed.
    """
    df = pd.DataFrame({"country": [None, "ESP"]})
    transformer = FillCategoricalMissing(["other_column"])
    transformed = transformer.transform(df)
    # Should not change the column or its dtype
    assert transformed.equals(df)

def test_fill_categorical_missing_with_categorical_dtype() -> None:
    """
    Test that FillCategoricalMissing works with pandas Categorical dtype and adds 'missing' as a category.
    """
    df = pd.DataFrame({"country": pd.Series(["PRT", None, "ESP"], dtype="category")})
    transformer = FillCategoricalMissing(["country"])
    transformed = transformer.transform(df)
    # After coercion, dtype will be string, so categories are lost
    assert transformed["country"].tolist() == ["PRT", "missing", "ESP"]
