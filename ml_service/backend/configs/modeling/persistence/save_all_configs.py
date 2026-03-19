"""Module for saving-specific code within the modeling configurations section of the ML service."""
from pathlib import Path

from ml_service.backend.configs.modeling.models.configs import ConfigPaths, ValidatedConfigs
from ml_service.backend.configs.persistence.save_config import save_config


def save_all_configs(validated_configs: ValidatedConfigs, paths: ConfigPaths) -> None:
    """Saves all modeling-related configurations to their respective YAML files.

    Args:
        validated_configs: The validated modeling configurations to be saved.
        paths: The paths to the YAML files where the configurations should be saved.
    """
    save_config(validated_configs.model_specs.model_dump(mode="json", exclude={"meta"}), Path(paths.model_specs))
    save_config(validated_configs.search.model_dump(mode="json"), Path(paths.search))
    save_config(validated_configs.training.model_dump(mode="json"), Path(paths.training))
