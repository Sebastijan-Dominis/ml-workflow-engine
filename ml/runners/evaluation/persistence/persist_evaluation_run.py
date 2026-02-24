import logging
from pathlib import Path

import pandas as pd

from ml.config.validation_schemas.model_cfg import TrainModelConfig
from ml.registry.hash_registry import hash_artifact
from ml.runners.evaluation.persistence.save_predictions import save_predictions
from ml.utils.experiments.persistence.save_metrics import save_metrics
from ml.utils.persistence.save_metadata import save_metadata
from ml.utils.runtime.save_runtime import save_runtime_snapshot
from ml.runners.evaluation.persistence.prepare_metadata import prepare_metadata

logger = logging.getLogger(__name__)

def persist_evaluation_run(
    model_cfg: TrainModelConfig, 
    *, 
    eval_run_id: str, 
    train_run_id: str, 
    experiment_dir: Path, 
    eval_run_dir: Path, 
    metrics: dict[str, dict[str, float]], 
    prediction_dfs: dict[str, pd.DataFrame], 
    feature_lineage: list[dict], 
    start_time: float, 
    timestamp: str, 
    artifacts: dict[str, str], 
    pipeline_cfg_hash: str
) -> None:
    metrics_file = save_metrics(
        metrics, 
        model_cfg=model_cfg, 
        target_run_id=eval_run_id, 
        experiment_dir=experiment_dir, 
        stage="evaluation"
    )
    artifacts["metrics_path"] = metrics_file
    artifacts["metrics_hash"] = hash_artifact(Path(metrics_file))

    predictions_paths = save_predictions(prediction_dfs, target_dir=eval_run_dir)
    for key, path in predictions_paths.items():
        artifacts[f"predictions_{key}_path"] = path
        artifacts[f"predictions_{key}_hash"] = hash_artifact(Path(path))

    metadata = prepare_metadata(
        model_cfg=model_cfg,
        eval_run_id=eval_run_id,
        train_run_id=train_run_id,
        experiment_dir=experiment_dir,
        feature_lineage=feature_lineage,
        artifacts=artifacts,
        pipeline_cfg_hash=pipeline_cfg_hash
    )

    save_metadata(metadata, target_dir=eval_run_dir)

    save_runtime_snapshot(
        target_dir=eval_run_dir, 
        timestamp=timestamp, 
        hardware_info=model_cfg.training.hardware, 
        start_time=start_time
    )