"""Unit tests for concrete feature-engineering operator transforms."""

import pandas as pd
import pytest
from ml.components.feature_engineering.adr_per_person import AdrPerPerson
from ml.components.feature_engineering.arrival_date import ArrivalDate
from ml.components.feature_engineering.arrival_season import ArrivalSeason
from ml.components.feature_engineering.total_stay import TotalStay

pytestmark = pytest.mark.unit


def test_arrival_season_maps_boundary_weeks_to_expected_labels() -> None:
    """Map seasonal boundaries consistently across ISO week cutoffs."""
    df = pd.DataFrame({"arrival_date_week_number": [1, 10, 21, 22, 34, 35, 47, 48, 52]})

    transformed = ArrivalSeason().transform(df)

    assert transformed["arrival_season"].tolist() == [
        "Winter",
        "Spring",
        "Spring",
        "Summer",
        "Summer",
        "Fall",
        "Fall",
        "Winter",
        "Winter",
    ]
    assert "arrival_season" not in df.columns


def test_arrival_date_builds_datetime_column_without_mutating_input() -> None:
    """Construct arrival_date from split date parts and leave source frame untouched."""
    df = pd.DataFrame(
        {
            "arrival_date_year": [2024, 2025],
            "arrival_date_month": ["February", "December"],
            "arrival_date_day_of_month": [29, 31],
        }
    )

    transformed = ArrivalDate().transform(df)

    assert transformed["arrival_date"].tolist() == [
        pd.Timestamp("2024-02-29"),
        pd.Timestamp("2025-12-31"),
    ]
    assert "arrival_date" not in df.columns


def test_arrival_date_maps_unrecognized_month_name_to_nat() -> None:
    """Produce `NaT` when month names are outside the hard-coded month mapping."""
    df = pd.DataFrame(
        {
            "arrival_date_year": [2024],
            "arrival_date_month": ["Feb"],
            "arrival_date_day_of_month": [10],
        }
    )

    transformed = ArrivalDate().transform(df)

    assert transformed["arrival_date"].isna().tolist() == [True]
    assert "arrival_date" not in df.columns


def test_arrival_date_sets_n_features_in_when_transform_called_before_fit() -> None:
    """Populate sklearn metadata when transform is called before explicit fit."""
    df = pd.DataFrame(
        {
            "arrival_date_year": [2024],
            "arrival_date_month": ["January"],
            "arrival_date_day_of_month": [1],
            "extra": [99],
        }
    )
    op = ArrivalDate()

    assert not hasattr(op, "n_features_in_")

    op.transform(df)

    assert op.n_features_in_ == 4


def test_adr_per_person_replaces_zero_party_size_denominator_with_one() -> None:
    """Avoid division-by-zero by coercing zero total guests to denominator one."""
    df = pd.DataFrame(
        {
            "adr": [100.0, 90.0],
            "adults": [0, 2],
            "children": [0, 1],
            "babies": [0, 0],
        }
    )

    transformed = AdrPerPerson().transform(df)

    assert transformed["adr_per_person"].tolist() == pytest.approx([100.0, 30.0])
    assert "adr_per_person" not in df.columns


def test_total_stay_sums_weekend_and_weeknight_columns() -> None:
    """Produce total stay nights as a simple additive composite feature."""
    df = pd.DataFrame(
        {
            "stays_in_weekend_nights": [0, 2, 1],
            "stays_in_week_nights": [3, 4, 0],
        }
    )

    transformed = TotalStay().transform(df)

    assert transformed["total_stay"].tolist() == [3, 6, 1]
    assert "total_stay" not in df.columns


def test_feature_operators_set_n_features_in_when_transform_called_before_fit() -> None:
    """Backfill sklearn compatibility metadata when transform is used directly first."""
    df = pd.DataFrame({"arrival_date_week_number": [10, 22], "extra": [1, 2]})
    op = ArrivalSeason()

    assert not hasattr(op, "n_features_in_")

    op.transform(df)

    assert op.n_features_in_ == 2
