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
    """Return the original series when target transformation is disabled."""
    y = pd.Series([1.0, 2.0, 3.0], name="target")
    config = TargetTransformConfig(enabled=False, type=None, lambda_value=None)

    result = transform_target(y, transform_config=config, split_name="train")

    assert result is y


def test_transform_target_log1p_preserves_index_and_name() -> None:
    """Apply `log1p` while preserving series index and name."""
    y = pd.Series([0.0, 1.0, 3.0], index=[10, 11, 12], name="adr")
    config = TargetTransformConfig(enabled=True, type="log1p", lambda_value=None)

    result = transform_target(y, transform_config=config, split_name="val")

    np.testing.assert_allclose(result.to_numpy(), np.log1p(y.to_numpy()))
    assert list(result.index) == [10, 11, 12]
    assert result.name == "adr"


def test_transform_target_sqrt_rejects_negative_values() -> None:
    """Reject `sqrt` transformation when target contains negative values."""
    y = pd.Series([1.0, -0.5, 2.0], name="target")
    config = TargetTransformConfig(enabled=True, type="sqrt", lambda_value=None)

    with pytest.raises(ConfigError, match="Sqrt transformation requires non-negative target values"):
        transform_target(y, transform_config=config, split_name="train")


def test_transform_target_sqrt_applies_elementwise_square_root() -> None:
    """Apply ``sqrt`` transformation for non-negative targets."""
    y = pd.Series([0.0, 1.0, 9.0], name="target")
    config = TargetTransformConfig(enabled=True, type="sqrt", lambda_value=None)

    result = transform_target(y, transform_config=config, split_name="train")

    np.testing.assert_allclose(result.to_numpy(), np.array([0.0, 1.0, 3.0]))


def test_transform_target_boxcox_rejects_non_positive_values() -> None:
    """Reject ``boxcox`` transform when targets include zero or negative values."""
    y = pd.Series([1.0, 0.0, 2.0], name="target")
    config = cast(TargetTransformConfig, type("Cfg", (), {"enabled": True, "type": "boxcox", "lambda_value": 0.5})())

    with pytest.raises(ConfigError, match="Box-Cox transformation requires strictly positive"):
        transform_target(y, transform_config=config, split_name="train")


def test_transform_target_boxcox_requires_lambda_value() -> None:
    """Reject ``boxcox`` transform when lambda is not configured."""
    y = pd.Series([1.0, 2.0, 3.0], name="target")
    config = cast(TargetTransformConfig, type("Cfg", (), {"enabled": True, "type": "boxcox", "lambda_value": None})())

    with pytest.raises(ConfigError, match="requires lambda_value"):
        transform_target(y, transform_config=config, split_name="train")


def test_inverse_transform_target_round_trip_for_sqrt() -> None:
    """Round-trip ``sqrt`` transform and inverse transform to original values."""
    y = pd.Series([0.0, 1.0, 4.0, 9.0], name="target")
    config = TargetTransformConfig(enabled=True, type="sqrt", lambda_value=None)

    transformed = transform_target(y, transform_config=config, split_name="train")
    restored = inverse_transform_target(transformed.to_numpy(), transform_config=config, split_name="train")

    np.testing.assert_allclose(restored, y.to_numpy())


def test_inverse_transform_target_round_trip_for_boxcox() -> None:
    """Round-trip ``boxcox`` transform and inverse transform to original values."""
    y = pd.Series([1.0, 2.0, 4.0, 8.0], name="target")
    config = cast(TargetTransformConfig, type("Cfg", (), {"enabled": True, "type": "boxcox", "lambda_value": 0.3})())

    transformed = transform_target(y, transform_config=config, split_name="train")
    restored = inverse_transform_target(transformed.to_numpy(), transform_config=config, split_name="train")

    np.testing.assert_allclose(restored, y.to_numpy(), rtol=1e-6, atol=1e-8)


def test_inverse_transform_target_boxcox_requires_lambda_value() -> None:
    """Reject inverse ``boxcox`` transform when lambda is missing."""
    arr = np.array([0.1, 0.2], dtype=float)
    config = cast(TargetTransformConfig, type("Cfg", (), {"enabled": True, "type": "boxcox", "lambda_value": None})())

    with pytest.raises(ConfigError, match="Box-Cox inverse requires lambda_value"):
        inverse_transform_target(arr, transform_config=config, split_name="val")


def test_transform_target_raises_on_unsupported_type() -> None:
    """Raise ``ConfigError`` for unsupported forward transform type."""
    y = pd.Series([1.0, 2.0], name="target")
    config = cast(TargetTransformConfig, type("Cfg", (), {"enabled": True, "type": "unsupported", "lambda_value": None})())

    with pytest.raises(ConfigError, match="Unsupported target transformation type"):
        transform_target(y, transform_config=config, split_name="val")


def test_inverse_transform_target_round_trip_for_log1p() -> None:
    """Round-trip `log1p` transform and inverse transform to original values."""
    y = pd.Series([0.0, 4.0, 9.0], name="target")
    config = TargetTransformConfig(enabled=True, type="log1p", lambda_value=None)

    transformed = transform_target(y, transform_config=config, split_name="train")
    restored = inverse_transform_target(transformed.to_numpy(), transform_config=config, split_name="train")

    np.testing.assert_allclose(restored, y.to_numpy())


def test_inverse_transform_target_returns_original_array_when_disabled() -> None:
    """Return the original array when inverse transform is disabled."""
    arr = np.array([0.1, 0.2, 0.3], dtype=float)
    config = TargetTransformConfig(enabled=False, type=None, lambda_value=None)

    result = inverse_transform_target(arr, transform_config=config, split_name="val")

    assert result is arr


def test_inverse_transform_target_raises_on_unsupported_type() -> None:
    """Raise `ConfigError` for unsupported inverse transform type."""
    arr = np.array([0.1, 0.2], dtype=float)
    config = cast(TargetTransformConfig, type("Cfg", (), {"enabled": True, "type": "unsupported", "lambda_value": None})())

    with pytest.raises(ConfigError, match="Unsupported target transformation type"):
        inverse_transform_target(arr, transform_config=config, split_name="val")
