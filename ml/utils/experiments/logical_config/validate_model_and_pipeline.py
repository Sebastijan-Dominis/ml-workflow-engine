"""Logical-config validation for model and pipeline artifact integrity."""

import logging
from pathlib import Path

from ml.exceptions import PipelineContractError
from ml.registry.hash_registry import hash_artifact
from ml.utils.loaders import load_json

logger = logging.getLogger(__name__)

def validate_model_and_pipeline(train_dir: Path) -> dict:
    """Validate persisted artifact hashes and return artifact metadata payload.

    Args:
        train_dir: Training run directory containing metadata and artifacts.

    Returns:
        Artifact metadata mapping after successful integrity checks.
    """

    train_metadata_file = train_dir / "metadata.json"
    train_metadata = load_json(train_metadata_file)

    artifacts = train_metadata.get("artifacts", {})

    if not artifacts:
        msg = f"No artifacts section found in training metadata at {train_metadata_file}"
        logger.error(msg)
        raise PipelineContractError(msg)

    pipeline_file = Path(artifacts.get("pipeline_path"))
    model_file = Path(artifacts.get("model_path"))

    model_hash = hash_artifact(model_file)
    pipeline_hash = hash_artifact(pipeline_file)

    expected_model_hash = artifacts.get("model_hash")
    expected_pipeline_hash = artifacts.get("pipeline_hash")

    if model_hash != expected_model_hash:
        msg = f"Model hash mismatch: expected {expected_model_hash}, got {model_hash}"
        logger.error(msg)
        raise PipelineContractError(msg)
    
    if pipeline_hash != expected_pipeline_hash:
        msg = f"Pipeline hash mismatch: expected {expected_pipeline_hash}, got {pipeline_hash}"
        logger.error(msg)
        raise PipelineContractError(msg)
    
    return artifacts