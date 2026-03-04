"""Preparation helpers for promotion run-info and metadata payloads."""

import argparse
import logging
from pathlib import Path

from ml.metadata.schemas.runners.explainability import ExplainabilityMetadata
from ml.metadata.schemas.runners.training import TrainingMetadata
from ml.promotion.config.models import PromotionThresholds
from ml.promotion.constants.constants import PreviousProductionRunIdentity
from ml.promotion.getters.get import get_pipeline_cfg_hash, get_training_conda_env_hash
from ml.utils.hashing.service import hash_thresholds
from ml.utils.runtime.runtime_snapshot import get_conda_env_export, hash_environment

logger = logging.getLogger(__name__)

def prepare_run_information(
    *,
    args: argparse.Namespace,
    experiment_id: str,
    train_run_id: str,
    eval_run_id: str,
    explain_run_id: str,
    run_id: str | None,
    timestamp: str,
    training_metadata: TrainingMetadata,
    explainability_metadata: ExplainabilityMetadata,
    metrics: dict,
    git_commit: str,
) -> dict:
    """Build run-info payload to be written into model registry entries.

    Args:
        args: Parsed promotion CLI arguments.
        experiment_id: Experiment identifier.
        train_run_id: Training run identifier.
        eval_run_id: Evaluation run identifier.
        explain_run_id: Explainability run identifier.
        run_id: Promotion/staging run identifier.
        timestamp: Current timestamp string.
        training_metadata: Training metadata payload.
        explainability_metadata: Explainability metadata payload.
        metrics: Evaluation metrics payload.
        git_commit: Git commit hash.

    Returns:
        dict: Registry-ready run information payload.
    """

    artifacts = explainability_metadata.artifacts

    feature_lineage = training_metadata.lineage.feature_lineage

    pipeline_cfg_hash = get_pipeline_cfg_hash(training_metadata)

    run_info = {
        "experiment_id": experiment_id,
        "train_run_id": train_run_id,
        "eval_run_id": eval_run_id,
        "explain_run_id": explain_run_id,
        "model_version": args.version,
        "pipeline_cfg_hash": pipeline_cfg_hash,

        "artifacts": artifacts.model_dump(),

        "feature_lineage": [f.model_dump() for f in feature_lineage],

        "metrics": metrics,

        "git_commit": git_commit,
    }

    if args.stage == "production":
        run_info["promotion_id"] = run_id
        run_info["promoted_at"] = timestamp
    elif args.stage == "staging":
        run_info["staging_id"] = run_id
        run_info["staged_at"] = timestamp

    return run_info

def prepare_metadata(
    *,
    run_id: str | None,
    args: argparse.Namespace,
    metrics: dict,
    previous_production_metrics: dict | None,
    promotion_thresholds: PromotionThresholds,
    promoted: bool,
    beats_previous: bool,
    reason: str,
    git_commit: str,
    timestamp: str,
    previous_production_run_identity: PreviousProductionRunIdentity,
    train_run_dir: Path
) -> dict:
    """Build promotion metadata payload persisted in promotion run directory.

    Args:
        run_id: Promotion/staging run identifier.
        args: Parsed promotion CLI arguments.
        metrics: Evaluation metrics payload.
        previous_production_metrics: Metrics for previous production run, if any.
        promotion_thresholds: Validated promotion thresholds.
        promoted: Whether promotion decision is positive.
        beats_previous: Whether candidate beats previous production model.
        reason: Human-readable decision reason.
        git_commit: Git commit hash.
        timestamp: Current timestamp string.
        previous_production_run_identity: Previous production run identity metadata.
        train_run_dir: Training run directory.

    Returns:
        dict: Promotion metadata payload.

    Notes:
        Promotion and training conda-environment hashes are compared and logged
        to surface reproducibility risk without blocking promotion flow.

    Side Effects:
        Captures runtime environment export and may emit warnings on
        environment/hash mismatches.
    """

    conda_env_export = get_conda_env_export()
    promotion_conda_env_hash = hash_environment(conda_env_export)

    training_conda_env_hash = get_training_conda_env_hash(train_run_dir)

    if promotion_conda_env_hash != training_conda_env_hash:
        msg = f"Conda environment hash for promotion process does not match conda environment hash for training run. Promotion conda env hash: {promotion_conda_env_hash}, Training conda env hash: {training_conda_env_hash}. This may indicate that the promotion process is running with a different conda environment than the training process, which could lead to inconsistencies and unexpected issues. Please ensure that the same conda environment is used for both training and promotion processes."
        logger.warning(msg)

    thresholds_hash = hash_thresholds(promotion_thresholds.model_dump())

    metadata = {
        "run_identity": {
            "experiment_id": args.experiment_id,
            "train_run_id": args.train_run_id,
            "eval_run_id": args.eval_run_id,
            "explain_run_id": args.explain_run_id,
            "stage": args.stage,
        },

        "previous_production_run_identity": {
            "experiment_id": previous_production_run_identity.experiment_id,
            "train_run_id": previous_production_run_identity.train_run_id,
            "eval_run_id": previous_production_run_identity.eval_run_id,
            "explain_run_id": previous_production_run_identity.explain_run_id,
            "promotion_id": previous_production_run_identity.promotion_id,
        },

        "metrics": metrics,

        "previous_production_metrics": previous_production_metrics,

        "promotion_thresholds": promotion_thresholds.model_dump(),
        "promotion_thresholds_hash": thresholds_hash,

        "decision": {
            "promoted": promoted,
            "reason": reason,
        },

        "context": {
            "git_commit": git_commit,
            "promotion_conda_env_hash": promotion_conda_env_hash,
            "training_conda_env_hash": training_conda_env_hash,
            "timestamp": timestamp
        }
    }

    if args.stage == "production":
        metadata["run_identity"]["promotion_id"] = run_id
        metadata["decision"]["beats_previous"] = beats_previous
    elif args.stage == "staging":
        metadata["run_identity"]["staging_id"] = run_id

    return metadata
