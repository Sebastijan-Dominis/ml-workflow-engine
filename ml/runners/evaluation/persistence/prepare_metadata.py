from pathlib import Path

from ml.config.validation_schemas.model_cfg import TrainModelConfig


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