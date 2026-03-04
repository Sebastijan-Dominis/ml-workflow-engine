"""Lineage validation helpers that verify runtime config hash consistency."""

import logging
from pathlib import Path
from typing import overload

from ml.config.hashing import compute_model_config_hash
from ml.config.schemas.model_cfg import SearchModelConfig, TrainModelConfig
from ml.exceptions import PipelineContractError
from ml.metadata.validation.runners.training import validate_training_metadata
from ml.utils.loaders import load_json

logger = logging.getLogger(__name__)

@overload
def validate_configs_match(train_dir: Path, cfg: TrainModelConfig) -> None:
    """Typed overload for validating training-config lineage hash consistency.

    Args:
        train_dir: Training run directory.
        cfg: Training configuration object.

    Returns:
        None: Raises on mismatch.
    """
    ...

@overload
def validate_configs_match(train_dir: Path, cfg: SearchModelConfig) -> None:
    """Typed overload for validating search-config lineage hash consistency.

    Args:
        train_dir: Training run directory.
        cfg: Search configuration object.

    Returns:
        None: Raises on mismatch.
    """
    ...

def validate_configs_match(train_dir: Path, cfg: TrainModelConfig | SearchModelConfig) -> None:
    """Validate that runtime config hash matches the hash persisted in metadata.

    Args:
        train_dir: Training run directory.
        cfg: Search or training configuration object.

    Returns:
        None: Raises on mismatch.
    """

    training_metadata_path = train_dir / "metadata.json"
    if not training_metadata_path.exists():
        msg = f"Lineage integrity validation failed: {training_metadata_path} does not exist."
        logger.error(msg)
        raise PipelineContractError(msg)
    training_metadata_raw = load_json(training_metadata_path)
    training_metadata = validate_training_metadata(training_metadata_raw)

    runtime_dict = cfg.model_dump(exclude={"_meta"}, by_alias=True)
    config_hash = compute_model_config_hash(runtime_dict)

    expected_config_hash = training_metadata.config_fingerprint.config_hash

    if not expected_config_hash:
        msg = (
            f"Lineage integrity validation failed: config hash not found in train metadata.\n"
            f"Train metadata path: {training_metadata_path}"
        )
        logger.error(msg)
        raise PipelineContractError(msg)

    if expected_config_hash != config_hash:
        msg = (
            f"Lineage integrity validation failed: config hash mismatch.\n"
            f"Expected hash: {expected_config_hash}\n"
            f"Actual hash: {config_hash}\n"
            f"Train metadata path: {training_metadata_path}"
        )
        logger.error(msg)
        raise PipelineContractError(msg)

    logger.debug("Lineage integrity validation passed: config hash matches.")
