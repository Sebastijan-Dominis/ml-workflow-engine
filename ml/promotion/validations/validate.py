import argparse
import logging
from pathlib import Path

from ml.exceptions import ConfigError, UserError
from ml.promotion.config.models import PromotionThresholds
from ml.promotion.constants.constants import RunnersMetadata

logger = logging.getLogger(__name__)

def validate_run_dirs(train_run_dir: Path, eval_run_dir: Path, explain_run_dir: Path) -> None:
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
    train_metadata = runners_metadata.train_metadata
    eval_metadata = runners_metadata.eval_metadata
    explain_metadata = runners_metadata.explain_metadata

    if eval_metadata.get("run_identity", {}).get("train_run_id") != train_metadata.get("run_identity", {}).get("train_run_id"):
        msg = f"Evaluation run {args.eval_run_id} is not linked to train run {args.train_run_id}"
        logger.error(msg)
        raise UserError(msg)
    
    if explain_metadata.get("run_identity", {}).get("train_run_id") != train_metadata.get("run_identity", {}).get("train_run_id"):
        msg = f"Explain run {args.explain_run_id} is not linked to train run {args.train_run_id}"
        logger.error(msg)
        raise UserError(msg)

def validate_explainability_artifacts(runners_metadata: RunnersMetadata, args: argparse.Namespace) -> None:
    explain_status = runners_metadata.explain_metadata.get("run_identity", {}).get("status")
    if explain_status != "success":
        msg = f"Explain run {args.explain_run_id} did not complete successfully. Status: {explain_status}"
        logger.error(msg)
        raise UserError(msg)

    train_artifacts = runners_metadata.train_metadata.get("artifacts", {})
    explain_artifacts = runners_metadata.explain_metadata.get("artifacts", {})

    if explain_artifacts.get("model_hash") is None:
        msg = f"Explain run {args.explain_run_id} is missing model hash artifact."
        logger.error(msg)
        raise UserError(msg)
    if explain_artifacts.get("model_hash") != train_artifacts.get("model_hash"):
        msg = f"Model hash in explain run {args.explain_run_id} does not match model hash in train run {args.train_run_id}."
        logger.error(msg)
        raise UserError(msg)
    
    if explain_artifacts.get("pipeline_hash") is None:
        msg = f"Explain run {args.explain_run_id} is missing pipeline hash artifact."
        logger.error(msg)
        raise UserError(msg)
    if explain_artifacts.get("pipeline_hash") != train_artifacts.get("pipeline_hash"):
        msg = f"Pipeline hash in explain run {args.explain_run_id} does not match pipeline hash in train run {args.train_run_id}."
        logger.error(msg)
        raise UserError(msg)
    
def validate_promotion_thresholds(promotion_thresholds: dict) -> PromotionThresholds:
    try:
        return PromotionThresholds(**promotion_thresholds)
    except Exception as e:
        msg = f"Invalid promotion thresholds configuration. Configuration: {promotion_thresholds}"
        logger.exception(msg)
        raise ConfigError(msg) from e