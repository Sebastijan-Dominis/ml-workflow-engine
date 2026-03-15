"""Module for loading-related code within modeling configurations section of the ML service."""
import yaml

from ml_service.backend.configs.modeling.formatting.timestamp import add_timestamp
from ml_service.backend.configs.modeling.models.configs import RawConfigsWithLineage


def load_all_yamls_and_add_lineage(payload: dict) -> RawConfigsWithLineage:
    """Load all YAML configs from the payload, add lineage information, and return them as a structured object.

    Args:
        payload (dict): A dictionary containing the YAML strings for model_specs, search, and training.

    Returns:
        RawConfigsWithLineage: A structured object containing the loaded configs with lineage information.
    """
    try:
        model_specs_yaml = payload.get("model_specs", "{}")
        search_yaml = payload.get("search", "{}")
        training_yaml = payload.get("training", "{}")

        model_specs_data = yaml.safe_load(model_specs_yaml) or {}
        search_data = yaml.safe_load(search_yaml) or {}
        training_data = yaml.safe_load(training_yaml) or {}

        if not model_specs_data or not search_data or not training_data:
            raise ValueError("One or more configs are empty or invalid YAML.")
    except yaml.YAMLError as e:
        raise ValueError(f"YAML parsing error: {str(e)}") from e

    model_specs_with_lineage = add_timestamp(model_specs_data, "model_specs_lineage")
    search_with_lineage = add_timestamp(search_data, "search_lineage")
    training_with_lineage = add_timestamp(training_data, "training_lineage")

    return RawConfigsWithLineage(
        model_specs=model_specs_with_lineage,
        search=search_with_lineage,
        training=training_with_lineage
    )
