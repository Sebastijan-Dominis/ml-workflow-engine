"""Validate feature freezing configuration."""

from ml.feature_freezing.freeze_strategies.tabular.config.models import TabularFeaturesConfig


def validate_feature_config(data: dict):
    """Validate config based on type."""

    cfg_type = data.get("type")

    if cfg_type == "tabular":
        return TabularFeaturesConfig(**data)

    raise ValueError(f"Unsupported feature config type: {cfg_type}")
