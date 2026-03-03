"""Metadata payload construction for evaluation runs."""

from pathlib import Path

from ml.config.schemas.model_cfg import TrainModelConfig


def prepare_metadata(
    model_cfg: TrainModelConfig, 
    *, 
    eval_run_id: str, 
    train_run_id: str, 
    experiment_dir: Path, 
    feature_lineage: list[dict], 
    artifacts: dict[str, str], 
    pipeline_cfg_hash: str
) -> dict:
    """Build evaluation metadata payload for persistence.

    Args:
        model_cfg: Validated training model configuration.
        eval_run_id: Evaluation run identifier.
        train_run_id: Source training run identifier.
        experiment_dir: Experiment directory path.
        feature_lineage: Feature lineage records for reproducibility.
        artifacts: Mapping of artifact names to persisted paths.
        pipeline_cfg_hash: Hash of the runtime pipeline configuration.

    Returns:
        Evaluation metadata dictionary for run persistence.
    """

    metadata = {
        "run_identity": {
            "stage": "evaluation",
            "eval_run_id": eval_run_id,
            "train_run_id": train_run_id,
            "snapshot_id": experiment_dir.name,
            "status": "success",
        },
        "lineage": {
            "feature_lineage": feature_lineage,
            "target_column": model_cfg.target.name,
            "problem": model_cfg.problem,
            "segment": model_cfg.segment.name,
            "model_version": model_cfg.version,
        },
        "config_fingerprint": {
            "config_hash": model_cfg.meta.config_hash,
            "pipeline_cfg_hash": pipeline_cfg_hash,
        },
        "artifacts": artifacts
    }
    return metadata