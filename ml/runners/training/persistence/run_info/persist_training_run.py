"""Persistence orchestration for training run metadata, metrics, and runtime."""

import logging
from pathlib import Path

from ml.config.validation_schemas.model_cfg import TrainModelConfig
from ml.utils.persistence.save_metadata import save_metadata
from ml.utils.experiments.persistence.save_metrics import save_metrics
from ml.utils.runtime.save_runtime import save_runtime_snapshot

logger = logging.getLogger(__name__)

def persist_training_run(
    model_cfg: TrainModelConfig, 
    *, 
    train_run_id: str, 
    experiment_dir: Path, 
    train_run_dir: Path, 
    start_time: float, 
    timestamp: str, 
    feature_lineage: list[dict], 
    metrics: dict[str, float], 
    model_hash: str, 
    pipeline_hash: str | None, 
    model_path: Path, 
    pipeline_path: Path | None, 
    pipeline_cfg_hash: str | None
) -> None:
    """Persist training metadata, metrics artifact, and runtime snapshot.

    Args:
        model_cfg: Validated training model configuration.
        train_run_id: Training run identifier.
        experiment_dir: Base experiment directory.
        train_run_dir: Training output directory.
        start_time: Process start time used for runtime metadata.
        timestamp: Run timestamp string.
        feature_lineage: Feature lineage records.
        metrics: Training metrics payload.
        model_hash: Trained model artifact hash.
        pipeline_hash: Optional pipeline artifact hash.
        model_path: Persisted model artifact path.
        pipeline_path: Optional persisted pipeline artifact path.
        pipeline_cfg_hash: Optional pipeline configuration hash.

    Returns:
        None.
    """

    metadata = {
        "run_identity": {
            "stage": "training",
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
        },
        "artifacts": {
            "model_hash": model_hash,
            "model_path": str(model_path),
        }
    }

    if pipeline_cfg_hash and pipeline_path and pipeline_hash:
        metadata["config_fingerprint"]["pipeline_cfg_hash"] = pipeline_cfg_hash
        metadata["artifacts"]["pipeline_path"] = str(pipeline_path)
        metadata["artifacts"]["pipeline_hash"] = pipeline_hash

    save_metadata(metadata, target_dir=train_run_dir)

    save_metrics(
        metrics, 
        model_cfg=model_cfg, 
        target_run_id=train_run_id, 
        experiment_dir=experiment_dir, 
        stage="training"
    )

    save_runtime_snapshot(
        target_dir=train_run_dir, 
        timestamp=timestamp, 
        hardware_info=model_cfg.training.hardware, 
        start_time=start_time
    )