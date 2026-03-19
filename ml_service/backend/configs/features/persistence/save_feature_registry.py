"""Atomic persistence for feature registry."""

import copy
from pathlib import Path

from ml.feature_freezing.freeze_strategies.tabular.config.models import TabularFeaturesConfig

from ml_service.backend.configs.features.utils.registry import load_registry
from ml_service.backend.configs.persistence.save_config import save_config


def save_feature_registry(
        name: str,
        version: str,
        *,
        validated_config: TabularFeaturesConfig,
        registry_path: Path
    ) -> dict:

    registry = load_registry(registry_path)

    new_registry = copy.deepcopy(registry)

    if name not in  new_registry:
        new_registry[name] = {}

    new_registry[name][version] = validated_config.model_dump(mode="json")

    registry_path.parent.mkdir(parents=True, exist_ok=True)

    save_config(new_registry, registry_path)

    return {
        "status": "written",
        "name": name,
        "version": version,
        "path": str(registry_path),
    }
