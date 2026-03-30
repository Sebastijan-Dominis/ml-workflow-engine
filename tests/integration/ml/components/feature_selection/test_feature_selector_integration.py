from __future__ import annotations

import pandas as pd
import pytest
from ml.components.feature_selection.selector import FeatureSelector
from ml.exceptions import DataError

pytestmark = pytest.mark.integration


def test_feature_selector_returns_only_selected_columns() -> None:
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    selector = FeatureSelector(selected_features=["a", "c"])

    out = selector.transform(df)

    assert list(out.columns) == ["a", "c"]
    assert out.shape == (2, 2)


def test_feature_selector_raises_on_missing_columns() -> None:
    df = pd.DataFrame({"a": [1]})
    selector = FeatureSelector(selected_features=["a", "b"])

    with pytest.raises(DataError):
        selector.transform(df)
