import logging
from pathlib import Path

import pandas as pd

from ml.config.validation_schemas.model_cfg import TrainModelConfig
from ml.runners.evaluation.persistence.save_predictions import save_predictions
from ml.utils.experiments.persistence.save_metrics import save_metrics
from ml.utils.persistence.save_metadata import save_metadata
from ml.utils.runtime.save_runtime import save_runtime_snapshot

logger = logging.getLogger(__name__)

def persist_evaluation_run(model_cfg: TrainModelConfig, *, eval_run_id: str, train_run_id: str, experiment_dir: Path, eval_run_dir: Path, metrics: dict[str, dict[str, float]], prediction_dfs: dict[str, pd.DataFrame], feature_lineage: list[dict], start_time: float, timestamp: str, artifacts: dict[str, str], pipeline_cfg_hash: str):
    metrics_file = save_metrics(
        metrics, 
        model_cfg=model_cfg, 
        target_run_id=eval_run_id, 
        experiment_dir=experiment_dir, 
        stage="evaluation"
    )

    predictions_paths = save_predictions(prediction_dfs, target_dir=eval_run_dir)

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
            "target_column": model_cfg.target,
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

    metadata["artifacts"]["metrics_path"] = metrics_file
    for key, path in predictions_paths.items():
        metadata["artifacts"][f"predictions_{key}_path"] = path

    save_metadata(metadata, target_dir=eval_run_dir)

    save_runtime_snapshot(
        target_dir=eval_run_dir, 
        timestamp=timestamp, 
        hardware_info=model_cfg.training.hardware, 
        start_time=start_time
    )