"""A module for validating pipeline configuration payloads against the PipelineConfig schema."""
from ml.pipelines.models import PipelineConfig


def validate_config_payload(payload: dict) -> PipelineConfig:
    """Validate the incoming payload against the PipelineConfig schema.

    Args:
        payload (dict): The incoming configuration payload to validate.

    Returns:
        PipelineConfig: The validated pipeline configuration object.
    """
    return PipelineConfig(**payload)
