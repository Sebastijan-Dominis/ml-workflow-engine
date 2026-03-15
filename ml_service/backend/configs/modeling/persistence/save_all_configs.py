import os

import yaml
from ml_service.backend.configs.modeling.models.configs import ConfigPaths, ValidatedConfigs


def save_all_configs(validated_configs: ValidatedConfigs, paths: ConfigPaths) -> None:
    os.makedirs(os.path.dirname(paths.model_specs), exist_ok=True)
    os.makedirs(os.path.dirname(paths.search), exist_ok=True)
    os.makedirs(os.path.dirname(paths.training), exist_ok=True)

    with open(paths.model_specs, "w") as f:
        yaml.safe_dump(
            validated_configs.model_specs.model_dump(mode="json", exclude={"meta"}),
            f,
            sort_keys=False,
            default_flow_style=False,
        )
    with open(paths.search, "w") as f:
        yaml.safe_dump(
            validated_configs.search.model_dump(mode="json"),
            f,
            sort_keys=False,
            default_flow_style=False,
        )
    with open(paths.training, "w") as f:
        yaml.safe_dump(
            validated_configs.training.model_dump(mode="json"),
            f,
            sort_keys=False,
            default_flow_style=False,
        )
