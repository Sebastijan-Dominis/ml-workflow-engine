"""Unit tests for categorical missing-value imputation transformer."""

from __future__ import annotations

import pandas as pd
import pytest
from ml.components.imputation.categorical import FillCategoricalMissing

pytestmark = pytest.mark.unit


def test_fill_categorical_missing_replaces_missing_values_and_preserves_input() -> None:
    """Impute configured categorical columns without mutating the source dataframe."""
    df = pd.DataFrame(
        {
            "country": ["PRT", None, "ESP"],
            "market_segment": ["Online TA", None, "Offline TA/TO"],
            "numeric": [1, 2, 3],
        }
    )
    transformer = FillCategoricalMissing(["country", "market_segment"])

    transformed = transformer.transform(df)

    assert transformed["country"].tolist() == ["PRT", "missing", "ESP"]
    assert transformed["market_segment"].tolist() == ["Online TA", "missing", "Offline TA/TO"]
    assert transformed["numeric"].tolist() == [1, 2, 3]
    assert df["country"].isna().tolist() == [False, True, False]
    assert df["market_segment"].isna().tolist() == [False, True, False]


def test_fill_categorical_missing_coerces_non_string_values_to_strings() -> None:
    """Coerce categorical values to strings as part of imputation preprocessing."""
    df = pd.DataFrame({"room_type": [1, 2, None]})
    transformer = FillCategoricalMissing(["room_type"])

    transformed = transformer.transform(df)

    assert transformed["room_type"].tolist() == ["1.0", "2.0", "missing"]


def test_fill_categorical_missing_fit_returns_self() -> None:
    """Return the transformer instance from `fit` because implementation is stateless."""
    transformer = FillCategoricalMissing(["country"])
    df = pd.DataFrame({"country": ["PRT"]})

    result = transformer.fit(df)

    assert result is transformer
