"""Logical-config validation for model and pipeline artifact integrity."""

import logging
from pathlib import Path

from ml.exceptions import PipelineContractError
from ml.metadata.validation.runners.training import validate_training_metadata
from ml.modeling.models.artifacts import Artifacts
from ml.utils.hashing.service import hash_artifact
from ml.utils.loaders import load_json

logger = logging.getLogger(__name__)

def validate_model_and_pipeline(train_dir: Path) -> Artifacts:
    """Validate persisted artifact hashes and return artifact metadata payload.

    Args:
        train_dir: Training run directory containing metadata and artifacts.

    Returns:
        Artifact metadata mapping after successful integrity checks.
    """

    training_metadata_file = train_dir / "metadata.json"
    training_metadata_raw = load_json(training_metadata_file)
    training_metadata = validate_training_metadata(training_metadata_raw)

    artifacts = training_metadata.artifacts

    model_file = Path(artifacts.model_path)
    model_hash = hash_artifact(model_file)
    expected_model_hash = artifacts.model_hash
    if model_hash != expected_model_hash:
        msg = f"Model hash mismatch: expected {expected_model_hash}, got {model_hash}"
        logger.error(msg)
        raise PipelineContractError(msg)

    if artifacts.pipeline_path:
        pipeline_file = Path(artifacts.pipeline_path)
        if pipeline_file.exists():
            pipeline_hash = hash_artifact(pipeline_file)
            expected_pipeline_hash = artifacts.pipeline_hash
            if pipeline_hash != expected_pipeline_hash:
                msg = f"Pipeline hash mismatch: expected {expected_pipeline_hash}, got {pipeline_hash}"
                logger.error(msg)
                raise PipelineContractError(msg)

    return artifacts
