"""A module for loading artifacts in the inference pipeline."""
from pathlib import Path

from ml.exceptions import PipelineContractError
from ml.post_promotion.inference.classes.function_returns import ArtifactLoadingReturn
from ml.promotion.config.registry_entry import RegistryEntry
from ml.runners.shared.loading.pipeline import load_model_or_pipeline
from ml.utils.hashing.service import hash_artifact


def load_and_validate_artifact(model_metadata: RegistryEntry) -> ArtifactLoadingReturn:
    """Load model or pipeline artifact based on provided metadata and validate its integrity.

    Args:
        model_metadata: Metadata containing artifact paths and expected hash.

    Returns:
        ArtifactLoadingReturn: The loaded and validated artifact.
    """
    pipeline_path = Path(model_metadata.artifacts.pipeline_path or "")
    model_path = Path(model_metadata.artifacts.model_path)
    expected_hash = model_metadata.artifacts.pipeline_hash

    if pipeline_path.exists():
        artifact = load_model_or_pipeline(pipeline_path, target_type="pipeline")
        actual_hash = hash_artifact(pipeline_path)
    elif model_path.exists():
        artifact = load_model_or_pipeline(model_path, target_type="model")
        actual_hash = hash_artifact(model_path)
    else:
        raise PipelineContractError("No valid artifact found.")

    if actual_hash != expected_hash:
        raise PipelineContractError(f"Hash mismatch! Expected {expected_hash}, got {actual_hash}")

    return ArtifactLoadingReturn(
        artifact=artifact,
        artifact_hash=actual_hash,
        artifact_type="pipeline" if pipeline_path.exists() else "model"
    )
