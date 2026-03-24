"""A module for validating config payloads."""
from ml.data.config.schemas.interim import InterimConfig
from ml.data.config.schemas.processed import ProcessedConfig


def validate_config_payload(config_type: str, payload: dict) -> InterimConfig | ProcessedConfig:
    """Validate payload with the correct schema.

    Args:
        config_type: "interim" or "processed"
        payload: Configuration payload to validate

    Returns:
        Validated config object
    """
    if config_type == "interim":
        return InterimConfig(**payload)
    elif config_type == "processed":
        return ProcessedConfig(**payload)
    else:
        raise ValueError(f"Unknown config_type: {config_type}")