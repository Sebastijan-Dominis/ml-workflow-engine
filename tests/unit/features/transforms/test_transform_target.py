"""Unit tests for the transform_target and inverse_transform_target functions."""
from typing import cast

import numpy as np
import pandas as pd
import pytest
from ml.config.schemas.model_specs import TargetTransformConfig
from ml.exceptions import ConfigError
from ml.features.transforms.transform_target import inverse_transform_target, transform_target

pytestmark = pytest.mark.unit


def test_transform_target_returns_original_series_when_disabled() -> None:
    """Test that when the target transformation is disabled, the original Series is returned unchanged."""
    y = pd.Series([1.0, 2.0, 3.0], name="target")
    config = TargetTransformConfig(enabled=False, type=None, lambda_value=None)

    result = transform_target(y, transform_config=config, split_name="train")

    assert result is y


def test_transform_target_log1p_preserves_index_and_name() -> None:
    """Test that the log1p transformation is applied correctly and that the index and name of the Series are preserved."""
    y = pd.Series([0.0, 1.0, 3.0], index=[10, 11, 12], name="adr")
    config = TargetTransformConfig(enabled=True, type="log1p", lambda_value=None)

    result = transform_target(y, transform_config=config, split_name="val")

    np.testing.assert_allclose(result.to_numpy(), np.log1p(y.to_numpy()))
    assert list(result.index) == [10, 11, 12]
    assert result.name == "adr"


def test_transform_target_sqrt_rejects_negative_values() -> None:
    """Test that the sqrt transformation raises a ConfigError when the target contains negative values."""
    y = pd.Series([1.0, -0.5, 2.0], name="target")
    config = TargetTransformConfig(enabled=True, type="sqrt", lambda_value=None)

    with pytest.raises(ConfigError, match="Sqrt transformation requires non-negative target values"):
        transform_target(y, transform_config=config, split_name="train")


def test_inverse_transform_target_round_trip_for_log1p() -> None:
    """Test that applying the log1p transformation and then the inverse transformation returns the original values."""
    y = pd.Series([0.0, 4.0, 9.0], name="target")
    config = TargetTransformConfig(enabled=True, type="log1p", lambda_value=None)

    transformed = transform_target(y, transform_config=config, split_name="train")
    restored = inverse_transform_target(transformed.to_numpy(), transform_config=config, split_name="train")

    np.testing.assert_allclose(restored, y.to_numpy())


def test_inverse_transform_target_returns_original_array_when_disabled() -> None:
    """Test that when the target transformation is disabled, the original array is returned unchanged."""
    arr = np.array([0.1, 0.2, 0.3], dtype=float)
    config = TargetTransformConfig(enabled=False, type=None, lambda_value=None)

    result = inverse_transform_target(arr, transform_config=config, split_name="val")

    assert result is arr


def test_inverse_transform_target_raises_on_unsupported_type() -> None:
    """Test that the inverse_transform_target function raises a ConfigError when an unsupported transformation type is specified."""
    arr = np.array([0.1, 0.2], dtype=float)
    config = cast(TargetTransformConfig, type("Cfg", (), {"enabled": True, "type": "unsupported", "lambda_value": None})())

    with pytest.raises(ConfigError, match="Unsupported target transformation type"):
        inverse_transform_target(arr, transform_config=config, split_name="val")
