"""Metadata payload construction for evaluation runs."""
import logging
from pathlib import Path

from ml.config.schemas.model_cfg import TrainModelConfig
from ml.metadata.schemas.runners.evaluation import EvaluationArtifacts
from ml.metadata.validation.runners.evaluation import validate_evaluation_metadata
from ml.modeling.models.feature_lineage import FeatureLineage

logger = logging.getLogger(__name__)

def prepare_metadata(
    model_cfg: TrainModelConfig,
    *,
    eval_run_id: str,
    train_run_id: str,
    experiment_dir: Path,
    feature_lineage: list[FeatureLineage],
    artifacts: EvaluationArtifacts,
    pipeline_cfg_hash: str
) -> dict:
    """Build evaluation metadata payload for persistence.

    Args:
        model_cfg: Validated training model configuration.
        eval_run_id: Evaluation run identifier.
        train_run_id: Source training run identifier.
        experiment_dir: Experiment directory path.
        feature_lineage: Feature lineage records for reproducibility.
        artifacts: Evaluation artifacts object.
        pipeline_cfg_hash: Hash of the runtime pipeline configuration.

    Returns:
        Evaluation metadata dictionary for run persistence.
    """

    metadata_raw = {
        "run_identity": {
            "stage": "evaluation",
            "eval_run_id": eval_run_id,
            "train_run_id": train_run_id,
            "snapshot_id": experiment_dir.name,
            "status": "success",
        },
        "lineage": {
            "feature_lineage": [f.model_dump() for f in feature_lineage],
            "target_column": model_cfg.target.name,
            "problem": model_cfg.problem,
            "segment": model_cfg.segment.name,
            "model_version": model_cfg.version,
        },
        "config_fingerprint": {
            "config_hash": model_cfg.meta.config_hash,
            "pipeline_cfg_hash": pipeline_cfg_hash,
        },
        "artifacts": artifacts.model_dump(exclude_none=True),
    }
    metadata = validate_evaluation_metadata(metadata_raw)

    return metadata.model_dump(exclude_none=True)
