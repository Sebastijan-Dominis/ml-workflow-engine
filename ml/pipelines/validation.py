"""Pipeline config validation."""

import logging
from typing import Any

from ml.exceptions import ConfigError
from ml.pipelines.models import PipelineConfig

logger = logging.getLogger(__name__)

def validate_pipeline_config(pipeline_cfg_raw: dict[str, Any]) -> PipelineConfig:
    """Validate pipeline config dictionary against the PipelineConfig Pydantic model.

    Args:
        pipeline_cfg_raw: Dictionary containing pipeline config to be validated.

    Returns:
        An instance of PipelineConfig if validation is successful.

    Raises:
        ConfigError: If the input dictionary does not conform to the PipelineConfig schema.
    """

    try:
        pipeline_cfg = PipelineConfig.model_validate(pipeline_cfg_raw)
        logger.debug("Pipeline config validation successful.")
        return pipeline_cfg
    except Exception as e:
        msg = "Pipeline config validation failed."
        logger.exception(msg)
        raise ConfigError(msg) from e
