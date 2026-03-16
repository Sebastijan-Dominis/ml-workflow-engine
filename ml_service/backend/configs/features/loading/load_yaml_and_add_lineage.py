import yaml
from ml_service.backend.configs.formatting.timestamp import add_timestamp


def load_yaml_and_add_lineage(yaml_text: str) -> dict:
    """Parse YAML and inject timestamp."""

    data = yaml.safe_load(yaml_text)

    if "lineage" not in data:
        raise ValueError("Missing 'lineage' section.")

    return add_timestamp(data, "lineage")
