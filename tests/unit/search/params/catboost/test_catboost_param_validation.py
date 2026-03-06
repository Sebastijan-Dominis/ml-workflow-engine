"""Unit tests for CatBoost parameter-value validation helpers."""

import pytest
from ml.exceptions import ConfigError
from ml.search.params.catboost.validation import validate_param_value

pytestmark = pytest.mark.unit


def test_validate_param_value_ignores_unknown_parameter_name() -> None:
    """Do nothing when validating a parameter without registered constraints."""
    validate_param_value("unknown_param", 123, task_type="CPU")


def test_validate_param_value_ignores_none_value() -> None:
    """Treat ``None`` values as intentionally unset and skip validation."""
    validate_param_value("depth", None, task_type="CPU")


def test_validate_param_value_rejects_zero_when_zero_not_allowed() -> None:
    """Raise ``ConfigError`` when zero is passed to non-zero hyperparameters."""
    with pytest.raises(ConfigError, match="depth cannot be zero"):
        validate_param_value("depth", 0, task_type="CPU")


def test_validate_param_value_rejects_negative_when_negative_not_allowed() -> None:
    """Raise ``ConfigError`` when negative values violate sign constraints."""
    with pytest.raises(ConfigError, match="learning_rate cannot be negative"):
        validate_param_value("learning_rate", -0.1, task_type="CPU")


def test_validate_param_value_rejects_values_below_minimum() -> None:
    """Raise ``ConfigError`` for values below configured lower bounds."""
    with pytest.raises(ConfigError, match=r"depth=0.5 < min allowed 1"):
        validate_param_value("depth", 0.5, task_type="CPU")


def test_validate_param_value_rejects_values_above_maximum() -> None:
    """Raise ``ConfigError`` for values above configured upper bounds."""
    with pytest.raises(ConfigError, match=r"depth=17 > max allowed 16"):
        validate_param_value("depth", 17, task_type="CPU")


def test_validate_param_value_accepts_gpu_border_count_allowed_values() -> None:
    """Allow only GPU-safe ``border_count`` choices from the supported set."""
    for value in [32, 64, 128, 254]:
        validate_param_value("border_count", value, task_type="GPU")


def test_validate_param_value_rejects_gpu_border_count_outside_allowed_values() -> None:
    """Raise ``ConfigError`` for unsupported GPU ``border_count`` values."""
    with pytest.raises(ConfigError, match=r"border_count has to be one of \[32, 64, 128, 254\]"):
        validate_param_value("border_count", 100, task_type="GPU")


def test_validate_param_value_rejects_gpu_colsample_bylevel_not_equal_to_one() -> None:
    """Raise ``ConfigError`` when GPU ``colsample_bylevel`` differs from 1.0."""
    with pytest.raises(ConfigError, match=r"colsample_bylevel has to be 1.0"):
        validate_param_value("colsample_bylevel", 0.9, task_type="GPU")


def test_validate_param_value_accepts_valid_values_for_cpu_and_gpu() -> None:
    """Accept representative in-range values on both CPU and GPU paths."""
    validate_param_value("depth", 6, task_type="CPU")
    validate_param_value("learning_rate", 0.1, task_type="CPU")
    validate_param_value("colsample_bylevel", 1.0, task_type="GPU")
