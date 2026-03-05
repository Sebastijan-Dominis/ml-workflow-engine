"""Unit tests for high-impact validators in model_specs schemas."""

import pytest
from ml.config.schemas.model_specs import (
    FeatureImportanceMethodConfig,
    ScoringConfig,
    SegmentationConfig,
    SegmentationFilter,
    SHAPMethodConfig,
    TaskConfig,
)
from ml.exceptions import ConfigError

pytestmark = pytest.mark.unit


def test_task_config_normalizes_task_type_to_lowercase() -> None:
    """Test that the TaskConfig schema normalizes the task type to lowercase.

    Args:
        None

    Returns:
        None
    """
    cfg = TaskConfig.model_validate({"type": "CLASSIFICATION", "subtype": "binary"})

    assert cfg.type == "classification"
    assert cfg.subtype == "binary"


def test_segmentation_config_rejects_filters_when_disabled() -> None:
    """Test that the SegmentationConfig schema raises a ConfigError if filters are provided when segmentation is disabled.

    Args:
        None

    Returns:
        None
    """
    with pytest.raises(ConfigError, match="Segmentation filters should not be provided"):
        SegmentationConfig(
            enabled=False,
            include_in_model=False,
            filters=[SegmentationFilter(column="hotel", op="eq", value="City Hotel")],
        )


def test_segmentation_config_rejects_include_in_model_when_disabled() -> None:
    """Test that the SegmentationConfig schema raises a ConfigError if include_in_model is True when segmentation is disabled.

    Args:
        None

    Returns:
        None
    """
    with pytest.raises(ConfigError, match="include_in_model should be False"):
        SegmentationConfig(enabled=False, include_in_model=True, filters=[])


def test_segmentation_config_rejects_missing_filters_when_enabled() -> None:
    """Test that the SegmentationConfig schema raises a ConfigError if filters are not provided when segmentation is enabled.

    Args:
        None

    Returns:
        None
    """
    with pytest.raises(ConfigError, match="Segmentation filters must be provided"):
        SegmentationConfig(enabled=True, include_in_model=False, filters=[])


def test_segmentation_config_accepts_enabled_with_filters() -> None:
    """Test that the SegmentationConfig schema accepts a valid configuration when segmentation is enabled with filters.

    Args:
        None

    Returns:
        None
    """
    cfg = SegmentationConfig(
        enabled=True,
        include_in_model=True,
        filters=[SegmentationFilter(column="hotel", op="eq", value="City Hotel")],
    )

    assert cfg.enabled is True
    assert cfg.include_in_model is True
    assert len(cfg.filters) == 1


def test_scoring_config_rejects_fixed_policy_without_fixed_metric() -> None:
    """Test that the ScoringConfig schema raises a ConfigError if fixed_metric is not provided when policy is set to 'fixed'.

    Args:
        None

    Returns:
        None
    """
    with pytest.raises(ConfigError, match="fixed_metric must be specified"):
        ScoringConfig(policy="fixed", fixed_metric=None, pr_auc_threshold=None)


def test_scoring_config_rejects_adaptive_binary_without_pr_auc_threshold() -> None:
    """Test that the ScoringConfig schema raises a ConfigError if pr_auc_threshold is not provided when policy is set to 'adaptive_binary'.

    Args:
        None

    Returns:
        None
    """
    with pytest.raises(ConfigError, match="pr_auc_threshold must be specified"):
        ScoringConfig(policy="adaptive_binary", fixed_metric="roc_auc", pr_auc_threshold=None)


def test_scoring_config_accepts_valid_fixed_policy_payload() -> None:
    """Test that the ScoringConfig schema accepts a valid configuration when policy is set to 'fixed'.

    Args:
        None

    Returns:
        None
    """
    cfg = ScoringConfig(policy="fixed", fixed_metric="roc_auc", pr_auc_threshold=None)

    assert cfg.policy == "fixed"
    assert cfg.fixed_metric == "roc_auc"


def test_feature_importance_method_rejects_enabled_without_type() -> None:
    """Test that the FeatureImportanceMethodConfig schema raises a ConfigError if type is not provided when feature importance method is enabled.

    Args:
        None

    Returns:
        None
    """
    with pytest.raises(ConfigError, match="Type must be specified if feature importance method is enabled"):
        FeatureImportanceMethodConfig(enabled=True, type=None)


def test_shap_method_rejects_enabled_without_approximate() -> None:
    """Test that the SHAPMethodConfig schema raises a ConfigError if approximate is not provided when SHAP method is enabled.

    Args:
        None

    Returns:
        None
    """
    with pytest.raises(ConfigError, match="Approximate method must be specified if SHAP method is enabled"):
        SHAPMethodConfig(enabled=True, approximate=None)
