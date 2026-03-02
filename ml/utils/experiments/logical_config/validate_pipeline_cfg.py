"""Validation helpers for pipeline configuration hash consistency."""

import logging
from pathlib import Path

from ml.config.hashing import compute_config_hash
from ml.config.validation_schemas.model_cfg import TrainModelConfig
from ml.exceptions import ConfigError, PipelineContractError
from ml.utils.loaders import load_json, load_yaml

logger = logging.getLogger(__name__)

def validate_pipeline_cfg(metadata_file: Path, model_cfg: TrainModelConfig) -> str:
    """Validate active pipeline config hash against metadata and return hash.

    Args:
        metadata_file: Metadata file path containing expected pipeline hash.
        model_cfg: Validated training configuration with pipeline location.

    Returns:
        Validated active pipeline configuration hash.
    """

    metadata = load_json(metadata_file)
    expected_pipeline_hash = metadata.get("metadata", {}).get("pipeline_hash")
    if not expected_pipeline_hash:
        expected_pipeline_hash = metadata.get("config_fingerprint", {}).get("pipeline_cfg_hash")
    if not expected_pipeline_hash:
        msg = f"Pipeline hash not found in metadata file {metadata_file} under 'metadata.pipeline_hash' or 'config_fingerprint.pipeline_cfg_hash'."
        logger.error(msg)
        raise ConfigError(msg)
    
    pipeline_path = Path(model_cfg.pipeline.path).resolve()
    if not pipeline_path.is_file():
        msg = f"Pipeline configuration file not found at {pipeline_path}."
        logger.error(msg)
        raise ConfigError(msg)
    
    pipeline_cfg = load_yaml(pipeline_path)
    actual_pipeline_hash = compute_config_hash(pipeline_cfg)

    if actual_pipeline_hash != expected_pipeline_hash:
        msg = f"Pipeline configuration hash mismatch. Expected: {expected_pipeline_hash}, Actual: {actual_pipeline_hash}. Please ensure the pipeline configuration matches the expected version."
        logger.error(msg)
        raise PipelineContractError(msg)
    
    logger.debug(f"Pipeline configuration hash validated successfully. Hash: {actual_pipeline_hash}")

    return actual_pipeline_hash