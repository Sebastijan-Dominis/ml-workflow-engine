"""Unit tests for fixed-column feature selection transformer."""

from __future__ import annotations

import pandas as pd
import pytest
from ml.components.feature_selection.selector import FeatureSelector
from ml.exceptions import DataError

pytestmark = pytest.mark.unit


def test_feature_selector_returns_selected_columns_in_requested_order() -> None:
    """Return only selected columns and preserve requested feature ordering."""
    df = pd.DataFrame(
        {
            "lead_time": [10, 20],
            "adr": [100.0, 80.0],
            "country": ["PRT", "ESP"],
        }
    )
    selector = FeatureSelector(["country", "adr"])

    transformed = selector.transform(df)

    assert transformed.columns.tolist() == ["country", "adr"]
    assert transformed["country"].tolist() == ["PRT", "ESP"]
    assert transformed["adr"].tolist() == [100.0, 80.0]


def test_feature_selector_raises_data_error_when_selected_column_missing() -> None:
    """Raise `DataError` naming missing columns when selection contract is violated."""
    df = pd.DataFrame({"lead_time": [10], "adr": [100.0]})
    selector = FeatureSelector(["lead_time", "country"])

    with pytest.raises(DataError, match=r"Missing columns: \['country'\]"):
        selector.transform(df)


def test_feature_selector_supports_empty_selection_list() -> None:
    """Return empty-column dataframe when selection list is intentionally empty."""
    df = pd.DataFrame({"lead_time": [10, 20], "adr": [100.0, 80.0]})
    selector = FeatureSelector([])

    transformed = selector.transform(df)

    assert transformed.shape == (2, 0)
    assert transformed.columns.tolist() == []
