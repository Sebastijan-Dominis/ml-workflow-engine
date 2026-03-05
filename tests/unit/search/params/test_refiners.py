"""Unit tests for search-parameter neighborhood refinement helpers."""

import pytest
from ml.exceptions import ConfigError
from ml.search.params.refiners import refine_border_count, refine_float_mult, refine_int

pytestmark = pytest.mark.unit


def test_refine_int_builds_deduplicated_bounded_neighborhood() -> None:
    """Generate a symmetric integer neighborhood, dedupe values, and enforce bounds."""
    result = refine_int(center=10, offsets=[0, 1, 1, 3, 20], low=8, high=13)

    assert result == [9, 10, 11, 13]


def test_refine_int_rejects_non_integer_center() -> None:
    """Raise `ConfigError` when `center` is not an integer."""
    with pytest.raises(ConfigError, match="Expected integer center value"):
        refine_int(center=10.5, offsets=[1, 2], low=0, high=100)


def test_refine_float_mult_rounds_deduplicates_and_clamps_to_bounds() -> None:
    """Round multiplied candidates, remove duplicates, and keep only in-range values."""
    result = refine_float_mult(
        center=0.1,
        factors=[1, 1.111111, 1.111114, 0.555555, 100.0],
        low=0.0,
        high=1.0,
        decimals=4,
    )

    assert result == [0.0556, 0.1, 0.1111]


def test_refine_float_mult_rejects_non_numeric_center() -> None:
    """Raise `ConfigError` when `center` is not numeric."""
    with pytest.raises(ConfigError, match="Expected numeric center value"):
        refine_float_mult(center="bad", factors=[0.9, 1.1], low=0.0, high=1.0)


@pytest.mark.parametrize(
    ("center", "expected"),
    [
        (32, [32, 64]),
        (64, [32, 64, 128]),
        (128, [64, 128, 254]),
        (254, [128, 254]),
    ],
)
def test_refine_border_count_returns_adjacent_allowed_values(center: int, expected: list[int]) -> None:
    """Return neighboring border_count options while respecting allowed boundaries."""
    assert refine_border_count(center) == expected


def test_refine_border_count_rejects_disallowed_value() -> None:
    """Raise `ConfigError` when `center` is outside the allowed CatBoost options."""
    with pytest.raises(ConfigError, match="is not in allowed options"):
        refine_border_count(63)
