"""Unit tests for the TargetTransformConfig schema."""
import pytest
from ml.config.schemas.model_specs import TargetTransformConfig
from ml.exceptions import ConfigError

pytestmark = pytest.mark.unit


def test_target_transform_config_defaults_are_safe() -> None:
    """Test that the default values of TargetTransformConfig are set to safe options that do not apply any transformation."""
    cfg = TargetTransformConfig()

    assert cfg.enabled is False
    assert cfg.type is None
    assert cfg.lambda_value is None


def test_target_transform_config_allows_log1p_without_lambda() -> None:
    """Test that the TargetTransformConfig allows the 'log1p' transformation without a lambda value."""
    cfg = TargetTransformConfig(enabled=True, type="log1p", lambda_value=None)

    assert cfg.enabled is True
    assert cfg.type == "log1p"


def test_target_transform_config_rejects_lambda_for_non_boxcox() -> None:
    """Test that the TargetTransformConfig rejects a lambda value for transformations other than 'boxcox'."""
    with pytest.raises(ConfigError, match="lambda_value should only be provided"):
        TargetTransformConfig(enabled=True, type="sqrt", lambda_value=0.3)
