"""Validation functions for ML promotion run metadata and directories."""

import argparse
import logging
from pathlib import Path

from ml.exceptions import UserError
from ml.promotion.constants.constants import RunnersMetadata

logger = logging.getLogger(__name__)

def validate_run_dirs(
    train_run_dir: Path,
    eval_run_dir: Path,
    explain_run_dir: Path
) -> None:
    """Validate required train/eval/explain run directories exist.

    Args:
        train_run_dir: Training run directory.
        eval_run_dir: Evaluation run directory.
        explain_run_dir: Explainability run directory.

    Returns:
        None: Raises on validation failure.
    """

    if not train_run_dir.exists():
        msg = f"Train run directory does not exist: {train_run_dir}"
        logger.error(msg)
        raise UserError(msg)
    if not eval_run_dir.exists():
        msg = f"Eval run directory does not exist: {eval_run_dir}"
        logger.error(msg)
        raise UserError(msg)
    if not explain_run_dir.exists():
        msg = f"Explain run directory does not exist: {explain_run_dir}"
        logger.error(msg)
        raise UserError(msg)

def validate_run_ids(
    *,
    args: argparse.Namespace,
    runners_metadata: RunnersMetadata
) -> None:
    """Validate eval/explain runs are linked to the provided train run.

    Args:
        args: Parsed CLI arguments with run IDs.
        runners_metadata: Loaded train/eval/explain metadata bundle.

    Returns:
        None: Raises on validation failure.
    """

    training_metadata = runners_metadata.training_metadata
    evaluation_metadata = runners_metadata.evaluation_metadata
    explainability_metadata = runners_metadata.explainability_metadata

    if evaluation_metadata.run_identity.train_run_id != training_metadata.run_identity.train_run_id:
        msg = f"Evaluation run {args.eval_run_id} is not linked to train run {args.train_run_id}"
        logger.error(msg)
        raise UserError(msg)

    if explainability_metadata.run_identity.train_run_id != training_metadata.run_identity.train_run_id:
        msg = f"Explain run {args.explain_run_id} is not linked to train run {args.train_run_id}"
        logger.error(msg)
        raise UserError(msg)
