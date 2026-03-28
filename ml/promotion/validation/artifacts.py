"""Validation functions for artifacts."""
import logging
from pathlib import Path

from ml.exceptions import UserError
from ml.metadata.schemas.runners.explainability import ExplainabilityArtifacts
from ml.promotion.constants.constants import RunnersMetadata
from ml.utils.hashing.service import hash_artifact

logger = logging.getLogger(__name__)


def validate_artifacts_consistency(
    runners_metadata: RunnersMetadata
) -> None:
    """Validate artifacts consistency across train/eval/explain runs.

    Args:
        runners_metadata: Loaded train/eval/explain metadata bundle.

    Returns:
        None: Raises on validation failure.
    """

    statuses = {
        "train": runners_metadata.training_metadata.run_identity.status,
        "eval": runners_metadata.evaluation_metadata.run_identity.status,
        "explain": runners_metadata.explainability_metadata.run_identity.status,
    }

    for name, status in statuses.items():
        if not status:
            msg = f"Missing run status for '{name}'. Statuses: {statuses}"
            logger.error(msg)
            raise UserError(msg)

        if status != "success":
            msg = f"Run '{name}' did not complete successfully. Statuses: {statuses}"
            logger.error(msg)
            raise UserError(msg)

    train_artifacts = runners_metadata.training_metadata.artifacts
    eval_artifacts = runners_metadata.evaluation_metadata.artifacts
    explain_artifacts = runners_metadata.explainability_metadata.artifacts

    model_paths = {
        "train": train_artifacts.model_path,
        "eval": eval_artifacts.model_path,
        "explain": explain_artifacts.model_path,
    }

    if len(set(model_paths.values())) != 1:
        msg = f"Model path mismatch across runs: {model_paths}"
        logger.error(msg)
        raise UserError(msg)

    model_path = model_paths["train"]
    if not model_path:
        msg = "Model path is missing from metadata."
        logger.error(msg)
        raise UserError(msg)

    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        msg = f"Model artifact path does not exist: {model_path}"
        logger.error(msg)
        raise UserError(msg)

    model_hash_runtime = hash_artifact(model_path_obj)

    model_hashes = {
        "runtime": model_hash_runtime,
        "train": train_artifacts.model_hash,
        "eval": eval_artifacts.model_hash,
        "explain": explain_artifacts.model_hash,
    }

    for name, h in model_hashes.items():
        if not h:
            msg = f"Missing model hash for '{name}'. Hashes: {model_hashes}"
            logger.error(msg)
            raise UserError(msg)

    if len(set(model_hashes.values())) != 1:
        msg = f"Model hash mismatch across runs/runtime. Hashes: {model_hashes}"
        logger.error(msg)
        raise UserError(msg)

    pipeline_paths = {
        "train": train_artifacts.pipeline_path,
        "eval": eval_artifacts.pipeline_path,
        "explain": explain_artifacts.pipeline_path,
    }

    pipeline_hashes = {
        "train": train_artifacts.pipeline_hash,
        "eval": eval_artifacts.pipeline_hash,
        "explain": explain_artifacts.pipeline_hash,
    }

    paths_present = [bool(p) for p in pipeline_paths.values()]
    hashes_present = [bool(h) for h in pipeline_hashes.values()]

    if len(set(paths_present)) > 1:
        msg = f"Inconsistent pipeline_path presence across runs: {pipeline_paths}"
        logger.error(msg)
        raise UserError(msg)

    if len(set(hashes_present)) > 1:
        msg = f"Inconsistent pipeline_hash presence across runs: {pipeline_hashes}"
        logger.error(msg)
        raise UserError(msg)

    if any(paths_present):
        if len(set(pipeline_paths.values())) != 1:
            msg = f"Pipeline path mismatch across runs: {pipeline_paths}"
            logger.error(msg)
            raise UserError(msg)

        pipeline_path = pipeline_paths["train"]
        if pipeline_path is None:
            msg = "Pipeline path unexpectedly None while marked present."
            logger.error(msg)
            raise UserError(msg)

        pipeline_path_obj = Path(pipeline_path)

        if not pipeline_path_obj.exists():
            msg = f"Pipeline artifact path does not exist: {pipeline_path}"
            logger.error(msg)
            raise UserError(msg)

        pipeline_hash_runtime = hash_artifact(pipeline_path_obj)

        full_pipeline_hashes = {
            "runtime": pipeline_hash_runtime,
            **pipeline_hashes,
        }

        for name, h in full_pipeline_hashes.items():
            if not h:
                msg = f"Missing pipeline hash for '{name}'. Hashes: {full_pipeline_hashes}"
                logger.error(msg)
                raise UserError(msg)

        if len(set(full_pipeline_hashes.values())) != 1:
            msg = f"Pipeline hash mismatch across runs/runtime. Hashes: {full_pipeline_hashes}"
            logger.error(msg)
            raise UserError(msg)

def validate_optional_artifact(
    path: str | None,
    expected_hash: str | None,
    name: str
) -> None:
    """Validate optional artifact presence and hash consistency.

    Args:
        path: Path to the artifact.
        expected_hash: Expected hash of the artifact.
        name: Name of the artifact for logging.

    Returns:
        None: Raises on validation failure.
    """
    if bool(path) != bool(expected_hash):
        msg = f"Inconsistent presence for '{name}'. Path: {path}, Hash: {expected_hash}"
        logger.error(msg)
        raise UserError(msg)

    if not path:
        return # Artifact not present, which is consistent

    path_obj = Path(path)

    if not path_obj.exists():
        msg = f"Explainability artifact does not exist: {path}"
        logger.error(msg)
        raise UserError(msg)

    actual_hash = hash_artifact(path_obj)

    if actual_hash != expected_hash:
        msg = (
            f"Artifact hash mismatch for '{name}'. "
            f"Expected: {expected_hash}, Actual: {actual_hash}"
        )
        logger.error(msg)
        raise UserError(msg)

def validate_explainability_artifacts(
    explainability_artifacts: ExplainabilityArtifacts
) -> None:
    """Validate explainability artifact paths and hashes.

    Args:
        explainability_artifacts: ExplainabilityArtifacts object containing paths and hashes.

    Returns:
        None: Raises on validation failure.
    """

    validate_optional_artifact(
        explainability_artifacts.top_k_feature_importances_path,
        explainability_artifacts.top_k_feature_importances_hash,
        "top_k_feature_importances",
    )

    validate_optional_artifact(
        explainability_artifacts.top_k_shap_importances_path,
        explainability_artifacts.top_k_shap_importances_hash,
        "top_k_shap_importances",
    )
