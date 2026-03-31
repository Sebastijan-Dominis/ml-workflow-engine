from __future__ import annotations

import pandas as pd
import pytest
from ml.components.imputation.categorical import FillCategoricalMissing

pytestmark = pytest.mark.integration


def test_fill_categorical_missing_handles_na_and_categorical() -> None:
    df = pd.DataFrame(
        {
            "col1": ["a", None, "b"],
            "cat_col": pd.Series(pd.Categorical(["x", None, "y"])),
        }
    )

    transformer = FillCategoricalMissing(categorical_features=["cat_col", "col1"])

    out = transformer.transform(df)

    # No missing values remain
    assert out["col1"].isnull().sum() == 0
    assert out["cat_col"].isnull().sum() == 0

    # 'missing' placeholder present and all values are strings
    assert "missing" in out["col1"].tolist()
    assert "missing" in out["cat_col"].tolist()
    assert all(isinstance(v, str) for v in out["col1"].tolist())
    assert all(isinstance(v, str) for v in out["cat_col"].tolist())
