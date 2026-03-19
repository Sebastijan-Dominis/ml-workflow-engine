"""A module to retrieve snapshot bindings from a registry based on a provided key."""

import logging
from functools import lru_cache
from pathlib import Path

from ml.snapshot_bindings.config.models import SnapshotBinding, SnapshotBindingsRegistry
from ml.snapshot_bindings.validation.validate_snapshot_binding import (
    validate_snapshot_binding,
    validate_snapshot_binding_registry,
)
from ml.utils.loaders import load_yaml

logger = logging.getLogger(__name__)

def get_and_validate_snapshot_binding(
        snapshot_binding_key: str,
        expect_dataset_bindings: bool = False,
        expect_feature_set_bindings: bool = False
    ) -> SnapshotBinding:
    """Retrieve and validate snapshot binding from registry based on the provided key.

    Args:
        snapshot_binding_key: Key to identify which snapshot binding to retrieve, typically defined in training metadata.
    """

    @lru_cache
    def _load_registry() -> SnapshotBindingsRegistry:
        raw = load_yaml(Path("configs/snapshot_bindings_registry/bindings.yaml"))
        return validate_snapshot_binding_registry(raw)

    validated_registry = _load_registry()
    snapshot_binding_config_raw = validated_registry.get(snapshot_binding_key)
    snapshot_binding_config = validate_snapshot_binding(
        snapshot_binding_config_raw,
        expect_dataset_bindings=expect_dataset_bindings,
        expect_feature_set_bindings=expect_feature_set_bindings
    )
    return snapshot_binding_config
