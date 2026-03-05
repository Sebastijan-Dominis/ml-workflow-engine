"""Unit tests for point-in-time-safe feature aggregation operator."""

import pandas as pd
import pytest
from ml.components.feature_engineering.pit_operator import PITOperator

pytestmark = pytest.mark.unit


def test_pit_operator_builds_shifted_group_history_without_leakage() -> None:
    """Use prior in-group history only and fill group-first rows with global mean."""
    df = pd.DataFrame(
        {
            "hotel": ["A", "A", "B", "B"],
            "arrival_datetime": pd.to_datetime(
                ["2024-01-02", "2024-01-01", "2024-01-01", "2024-01-03"]
            ),
            "adr": [20.0, 10.0, 30.0, 50.0],
        }
    )
    op = PITOperator(
        groupby_cols=["hotel"],
        agg_col="adr",
        agg_func="mean",
        feature_name="adr_hist_mean",
    )

    transformed = op.transform(df)

    assert transformed["hotel"].tolist() == ["A", "A", "B", "B"]
    assert transformed["arrival_datetime"].tolist() == [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-03"),
    ]
    assert transformed["adr_hist_mean"].tolist() == pytest.approx([27.5, 10.0, 27.5, 30.0])
    assert "adr_hist_mean" not in df.columns


def test_pit_operator_sets_n_features_in_when_transform_called_before_fit() -> None:
    """Backfill sklearn compatibility metadata when transform runs before explicit fit."""
    df = pd.DataFrame(
        {
            "segment": ["x", "x"],
            "arrival_datetime": pd.to_datetime(["2025-05-01", "2025-05-02"]),
            "value": [1.0, 2.0],
        }
    )
    op = PITOperator(
        groupby_cols=["segment"],
        agg_col="value",
        agg_func="sum",
        feature_name="value_hist_sum",
    )

    assert not hasattr(op, "n_features_in_")

    transformed = op.transform(df)

    assert op.n_features_in_ == 3
    assert op.output_features == ["value_hist_sum"]
    assert "value_hist_sum" in transformed.columns
