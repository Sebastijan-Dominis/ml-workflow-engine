"""Data extraction helpers for promotion workflow inputs and artifacts."""

import logging
from pathlib import Path

from ml.exceptions import PersistenceError, UserError
from ml.promotion.constants.constants import RunnersMetadata
from ml.utils.loaders import load_json

logger = logging.getLogger(__name__)

def get_runners_metadata(train_run_dir: Path, eval_run_dir: Path, explain_run_dir: Path) -> RunnersMetadata:
    """Load train/eval/explain metadata payloads from run directories.

    Args:
        train_run_dir: Training run directory.
        eval_run_dir: Evaluation run directory.
        explain_run_dir: Explainability run directory.

    Returns:
        RunnersMetadata: Wrapper containing loaded metadata payloads.
    """

    train_metadata = load_json(train_run_dir / "metadata.json")
    eval_metadata = load_json(eval_run_dir / "metadata.json")
    explain_metadata = load_json(explain_run_dir / "metadata.json")
    return RunnersMetadata(train_metadata, eval_metadata, explain_metadata)



def extract_thresholds(promotion_thresholds: dict, problem: str, segment: str) -> dict:
    """Extract problem/segment-specific threshold config from global mapping.

    Args:
        promotion_thresholds: Full promotion-threshold mapping.
        problem: Problem key.
        segment: Segment key.

    Returns:
        dict: Threshold configuration for the given problem and segment.
    """

    promotion_thresholds = promotion_thresholds.get(problem, {}).get(segment, {})
    if not promotion_thresholds:
        msg = f"No promotion thresholds found for problem={problem} segment={segment}"
        logger.error(msg)
        raise UserError(msg)
    
    return promotion_thresholds

def get_artifacts(explain_metadata: dict) -> dict:
    """Return required model artifact references from explainability metadata.

    Args:
        explain_metadata: Explainability metadata payload.

    Returns:
        dict: Required artifact metadata for promotion.
    """

    artifacts = explain_metadata.get("artifacts", {})

    if not artifacts or artifacts.get("model_hash") is None or artifacts.get("model_path") is None:
        msg = f"Explainability metadata is missing required artifact information. Artifacts found: {artifacts}"
        logger.error(msg)
        raise PersistenceError(msg)
    
    return artifacts

def get_feature_lineage(training_metadata: dict) -> list[str]:
    """Extract feature lineage information from training metadata.

    Args:
        training_metadata: Training metadata payload.

    Returns:
        list[str]: Feature lineage entries.
    """

    feature_lineage = training_metadata.get("lineage", {}).get("feature_lineage")
    if not feature_lineage:
        msg = "Training metadata is missing feature lineage information."
        logger.error(msg)
        raise PersistenceError(msg)
    return feature_lineage

def get_pipeline_cfg_hash(training_metadata: dict) -> str:
    """Extract pipeline config hash from training metadata.

    Args:
        training_metadata: Training metadata payload.

    Returns:
        str: Pipeline configuration hash.
    """

    pipeline_cfg_hash = training_metadata.get("config_fingerprint", {}).get("pipeline_cfg_hash")
    if not pipeline_cfg_hash:
        msg = "Training metadata is missing pipeline configuration hash information."
        logger.error(msg)
        raise PersistenceError(msg)
    return pipeline_cfg_hash

def get_training_conda_env_hash(train_run_dir: Path) -> str:
    """Load training runtime and return recorded conda environment hash.

    Args:
        train_run_dir: Training run directory.

    Returns:
        str: Conda environment hash captured during training.
    """

    training_runtime_file = train_run_dir / "runtime.json"
    training_runtime = load_json(training_runtime_file)
    training_conda_env_hash = training_runtime.get("environment", {}).get("conda_env_hash")
    if not training_conda_env_hash:
        msg = f"Training runtime information is missing conda environment hash. Runtime file: {training_runtime_file}"
        logger.error(msg)
        raise PersistenceError(msg)
    return training_conda_env_hash