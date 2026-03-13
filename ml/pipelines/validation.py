"""Pipeline config validation."""

import logging
from typing import Any

from ml.exceptions import ConfigError
from ml.metadata.validation.search.search import validate_search_record
from ml.pipelines.models import PipelineConfig
from ml.utils.loaders import load_json

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

def validate_pipeline_config_consistency(actual_hash, search_dir):
    search_metadata_raw = load_json(search_dir / "metadata.json")
    search_metadata = validate_search_record(search_metadata_raw)
    expected_hash = search_metadata.metadata.pipeline_hash
    if actual_hash != expected_hash:
        msg = (
            f"Pipeline config hash mismatch: actual {actual_hash} vs expected {expected_hash} "
            f"from search metadata in {search_dir}"
        )
        logger.error(msg)
        raise ConfigError(msg)
    else:
        logger.debug("Pipeline config hash matches search metadata.")
