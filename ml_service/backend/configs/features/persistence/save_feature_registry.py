"""Atomic persistence for feature registry."""

import tempfile
from pathlib import Path

import yaml
from ml.feature_freezing.freeze_strategies.tabular.config.models import TabularFeaturesConfig
from ml_service.backend.configs.features.utils.registry import load_registry


def save_feature_registry(
        name: str,
        version: str,
        *,
        validated_config: TabularFeaturesConfig,
        registry_path: Path
    ) -> dict:

    registry = load_registry(registry_path)

    if name not in registry:
        registry[name] = {}

    registry[name][version] = validated_config.model_dump(mode="json")

    registry_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        "w",
        delete=False,
        dir=registry_path.parent,
    ) as tmp:

        yaml.safe_dump(registry, tmp, sort_keys=False)

        tmp_path = Path(tmp.name)

    tmp_path.replace(registry_path)

    return {
        "status": "written",
        "name": name,
        "version": version,
        "path": str(registry_path),
    }
