"""A module for validating snapshot binding configurations."""
import logging

from ml.exceptions import ConfigError
from ml.snapshot_bindings.config.models import SnapshotBinding, SnapshotBindingsRegistry

logger = logging.getLogger(__name__)

def validate_snapshot_binding(
        snapshot_binding_config: SnapshotBinding | None,
        expect_dataset_bindings: bool = False,
        expect_feature_set_bindings: bool = False
    ) -> SnapshotBinding:
    """Validate the snapshot binding configuration against the defined data model.

    Args:
        snapshot_binding_config: The snapshot binding configuration to validate, typically loaded from a YAML file.
        expect_dataset_bindings: Whether to expect dataset bindings in the configuration.
        expect_feature_set_bindings: Whether to expect feature set bindings in the configuration.

    Returns:
        SnapshotBinding: The validated snapshot binding configuration as a Pydantic model instance.
    """

    if snapshot_binding_config is None:
        msg = "No snapshot binding configuration provided."
        logger.error(msg)
        raise ConfigError(msg)
    if expect_dataset_bindings and not snapshot_binding_config.datasets:
        msg = "Expected dataset bindings in snapshot binding configuration, but none were found."
        logger.error(msg)
        raise ConfigError(msg)
    if expect_feature_set_bindings and not snapshot_binding_config.feature_sets:
        msg = "Expected feature set bindings in snapshot binding configuration, but none were found."
        logger.error(msg)
        raise ConfigError(msg)
    return snapshot_binding_config

def validate_snapshot_binding_registry(snapshot_bindings_registry_config: dict) -> SnapshotBindingsRegistry:
    """Validate the snapshot bindings registry configuration against the defined data model.

    Args:
        snapshot_bindings_registry_config: The snapshot bindings registry configuration to validate, typically loaded from a YAML file.

    Returns:
        SnapshotBindingsRegistry: The validated snapshot bindings registry configuration as a Pydantic model instance.
    """

    try:
        validated_registry = SnapshotBindingsRegistry.model_validate(snapshot_bindings_registry_config)
        return validated_registry
    except Exception as e:
        msg = "Invalid snapshot bindings registry configuration."
        logger.exception(msg)
        raise ConfigError(msg) from e
