"""Unit tests for default narrowing-parameter utility helpers."""

from types import SimpleNamespace

import pytest
from ml.exceptions import ConfigError
from ml.search.params.utils import get_default_float_params, get_default_int_params

pytestmark = pytest.mark.unit


def test_get_default_int_params_prefers_explicit_non_empty_values() -> None:
    """Use configured offsets and bounds when values are explicitly set and truthy."""
    param_cfg = SimpleNamespace(offsets=[2, 5], low=3, high=99)

    result = get_default_int_params(param_cfg, default_offsets=[1], default_low=1, default_high=10)

    assert result == ([2, 5], 3, 99)


def test_get_default_int_params_falls_back_when_values_are_none_or_empty() -> None:
    """Fallback to defaults when config values are absent or empty lists."""
    param_cfg = SimpleNamespace(offsets=[], low=None, high=None)

    result = get_default_int_params(param_cfg, default_offsets=[1, 3], default_low=2, default_high=8)

    assert result == ([1, 3], 2, 8)


def test_get_default_int_params_treats_zero_bounds_as_falsy_and_uses_defaults() -> None:
    """Preserve current falsy-handling semantics where zero-valued bounds revert to defaults."""
    param_cfg = SimpleNamespace(offsets=[1], low=0, high=0)

    result = get_default_int_params(param_cfg, default_offsets=[2], default_low=4, default_high=12)

    assert result == ([1], 4, 12)


def test_get_default_float_params_prefers_explicit_values_and_decimals() -> None:
    """Use configured factors, bounds, and decimals when provided and valid."""
    param_cfg = SimpleNamespace(factors=[0.9, 1.1], low=0.01, high=5.0, decimals=4)

    result = get_default_float_params(
        param_cfg,
        default_factors=[0.8, 1.2],
        default_low=0.1,
        default_high=2.0,
        default_decimals=3,
    )

    assert result == ([0.9, 1.1], 0.01, 5.0, 4)


def test_get_default_float_params_falls_back_for_empty_inputs_and_missing_decimals() -> None:
    """Fallback to defaults when factors are empty and decimals are not provided."""
    param_cfg = SimpleNamespace(factors=[], low=None, high=None, decimals=None)

    result = get_default_float_params(
        param_cfg,
        default_factors=[0.7, 1.3],
        default_low=0.2,
        default_high=3.0,
        default_decimals=5,
    )

    assert result == ([0.7, 1.3], 0.2, 3.0, 5)


def test_get_default_float_params_rejects_non_positive_decimals() -> None:
    """Raise ConfigError when decimals is zero or negative."""
    param_cfg = SimpleNamespace(factors=[1.0], low=0.1, high=1.0, decimals=0)

    with pytest.raises(ConfigError, match="must be a positive integer"):
        get_default_float_params(
            param_cfg,
            default_factors=[0.9, 1.1],
            default_low=0.1,
            default_high=1.0,
            default_decimals=3,
        )
