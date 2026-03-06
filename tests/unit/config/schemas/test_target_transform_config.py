"""Unit tests for the TargetTransformConfig schema."""
import pytest
from ml.config.schemas.model_specs import TargetTransformConfig
from ml.exceptions import ConfigError

pytestmark = pytest.mark.unit


def test_target_transform_config_defaults_are_safe() -> None:
    """Default target transform settings to a no-op configuration."""
    cfg = TargetTransformConfig()

    assert cfg.enabled is False
    assert cfg.type is None
    assert cfg.lambda_value is None


def test_target_transform_config_allows_log1p_without_lambda() -> None:
    """Allow `log1p` transform without requiring `lambda_value`."""
    cfg = TargetTransformConfig(enabled=True, type="log1p", lambda_value=None)

    assert cfg.enabled is True
    assert cfg.type == "log1p"


def test_target_transform_config_rejects_lambda_for_non_boxcox() -> None:
    """Reject `lambda_value` for non-BoxCox transforms."""
    with pytest.raises(ConfigError, match="lambda_value should only be provided"):
        TargetTransformConfig(enabled=True, type="sqrt", lambda_value=0.3)
