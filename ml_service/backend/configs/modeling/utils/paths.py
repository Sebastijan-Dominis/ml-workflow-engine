"""A module for path computations and checks related to model specs, search, and training configurations."""
import os

from ml_service.backend.configs.modeling.models.configs import ConfigPaths, ValidatedConfigs

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
env = os.environ.copy()
env["PYTHONPATH"] = repo_root

def compute_paths(validated_configs: ValidatedConfigs) -> ConfigPaths:
    """Computes the file paths for model specs, search, and training configurations based on the validated configurations provided.

    Args:
        validated_configs (ValidatedConfigs): The validated configurations containing model specifications and other relevant information.

    Returns:
        ConfigPaths: The computed file paths for each configuration type.
    """
    model_specs = validated_configs.model_specs

    model_specs_path = f"{repo_root}/configs/model_specs/{model_specs.problem}/{model_specs.segment.name}/{model_specs.version}.yaml"
    search_path = f"{repo_root}/configs/search/{model_specs.problem}/{model_specs.segment.name}/{model_specs.version}.yaml"
    training_path = f"{repo_root}/configs/train/{model_specs.problem}/{model_specs.segment.name}/{model_specs.version}.yaml"

    return ConfigPaths(
        model_specs=model_specs_path,
        search=search_path,
        training=training_path
    )

def check_paths(validated_configs: ValidatedConfigs) -> ConfigPaths:
    """Checks if the computed paths for model specs, search, and training configurations already exist.

    Args:
        validated_configs (ValidatedConfigs): The validated configurations containing model specifications and other relevant information.

    Returns:
        ConfigPaths: The computed file paths for each configuration type.
    """
    paths = compute_paths(validated_configs)

    if os.path.exists(paths.model_specs):
        raise FileExistsError(f"Model specs config already exists at {paths.model_specs}\nOverwriting existing configs is not allowed.")
    if os.path.exists(paths.search):
        raise FileExistsError(f"Search config already exists at {paths.search}\nOverwriting existing configs is not allowed.")
    if os.path.exists(paths.training):
        raise FileExistsError(f"Training config already exists at {paths.training}\nOverwriting existing configs is not allowed.")

    return paths
