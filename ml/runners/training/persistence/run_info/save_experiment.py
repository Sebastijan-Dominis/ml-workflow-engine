# General imports
import logging
import time
from pathlib import Path

from ml.config.validation_schemas.model_cfg import TrainModelConfig
from ml.runners.training.persistence.run_info.save_metadata import save_metadata
from ml.runners.training.persistence.run_info.save_metrics import save_metrics
from ml.utils.runtime.save_runtime import save_runtime_snapshot

logger = logging.getLogger(__name__)

def save_experiment(model_cfg: TrainModelConfig, *, train_run_id: str, experiment_dir: Path, start_time: float, timestamp: str, feature_lineage: list[dict], metrics: dict[str, float], model_hash: str, pipeline_hash: str | None, model_path: str, pipeline_path: str | None, pipeline_cfg_hash: str | None) -> None:
    metadata = {
        "run_identity": {
            "stage": "training",
            "train_run_id": train_run_id,
            "snapshot_id": experiment_dir.name,
            "status": "success",
        },
        "lineage": {
            "feature_lineage": feature_lineage,
            "target_column": model_cfg.target,
            "problem": model_cfg.problem,
            "segment": model_cfg.segment.name,
            "model_version": model_cfg.version,
        },
        "config_fingerprint": {
            "config_hash": model_cfg.meta.config_hash,
        },
        "artifacts": {
            "model_hash": model_hash,
            "model_path": model_path,
        }
    }

    if pipeline_cfg_hash and pipeline_path and pipeline_hash:
        metadata["config_fingerprint"]["pipeline_cfg_hash"] = pipeline_cfg_hash
        metadata["artifacts"]["pipeline_path"] = pipeline_path
        metadata["artifacts"]["pipeline_hash"] = pipeline_hash

    save_metadata(metadata, train_run_id=train_run_id, experiment_dir=experiment_dir)

    train_json = {
        "task_type": model_cfg.task.type,
        "algorithm": model_cfg.algorithm.value,
        "metrics": metrics,
    }

    save_metrics(train_json, train_run_id=train_run_id, experiment_dir=experiment_dir)

    runtime_dir = experiment_dir / "training" / train_run_id
    save_runtime_snapshot(runtime_dir, timestamp=timestamp, hardware_info=model_cfg.training.hardware, start_time=start_time)