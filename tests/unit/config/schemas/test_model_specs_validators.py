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
    """Normalize task type values to lowercase."""
    cfg = TaskConfig.model_validate({"type": "CLASSIFICATION", "subtype": "binary"})

    assert cfg.type == "classification"
    assert cfg.subtype == "binary"


def test_segmentation_config_rejects_filters_when_disabled() -> None:
    """Reject segmentation filters when segmentation is disabled."""
    with pytest.raises(ConfigError, match="Segmentation filters should not be provided"):
        SegmentationConfig(
            enabled=False,
            include_in_model=False,
            filters=[SegmentationFilter(column="hotel", op="eq", value="City Hotel")],
        )


def test_segmentation_config_rejects_include_in_model_when_disabled() -> None:
    """Reject `include_in_model=True` when segmentation is disabled."""
    with pytest.raises(ConfigError, match="include_in_model should be False"):
        SegmentationConfig(enabled=False, include_in_model=True, filters=[])


def test_segmentation_config_rejects_missing_filters_when_enabled() -> None:
    """Require filters when segmentation is enabled."""
    with pytest.raises(ConfigError, match="Segmentation filters must be provided"):
        SegmentationConfig(enabled=True, include_in_model=False, filters=[])


def test_segmentation_config_accepts_enabled_with_filters() -> None:
    """Accept enabled segmentation with a valid filter list."""
    cfg = SegmentationConfig(
        enabled=True,
        include_in_model=True,
        filters=[SegmentationFilter(column="hotel", op="eq", value="City Hotel")],
    )

    assert cfg.enabled is True
    assert cfg.include_in_model is True
    assert len(cfg.filters) == 1


def test_scoring_config_rejects_fixed_policy_without_fixed_metric() -> None:
    """Require `fixed_metric` for fixed scoring policy."""
    with pytest.raises(ConfigError, match="fixed_metric must be specified"):
        ScoringConfig(policy="fixed", fixed_metric=None, pr_auc_threshold=None)


def test_scoring_config_rejects_adaptive_binary_without_pr_auc_threshold() -> None:
    """Require `pr_auc_threshold` for adaptive-binary scoring policy."""
    with pytest.raises(ConfigError, match="pr_auc_threshold must be specified"):
        ScoringConfig(policy="adaptive_binary", fixed_metric="roc_auc", pr_auc_threshold=None)


def test_scoring_config_accepts_valid_fixed_policy_payload() -> None:
    """Accept valid fixed-policy scoring configuration."""
    cfg = ScoringConfig(policy="fixed", fixed_metric="roc_auc", pr_auc_threshold=None)

    assert cfg.policy == "fixed"
    assert cfg.fixed_metric == "roc_auc"


def test_feature_importance_method_rejects_enabled_without_type() -> None:
    """Require feature-importance type when method is enabled."""
    with pytest.raises(ConfigError, match="Type must be specified if feature importance method is enabled"):
        FeatureImportanceMethodConfig(enabled=True, type=None)


def test_shap_method_rejects_enabled_without_approximate() -> None:
    """Require `approximate` setting when SHAP method is enabled."""
    with pytest.raises(ConfigError, match="Approximate method must be specified if SHAP method is enabled"):
        SHAPMethodConfig(enabled=True, approximate=None)
