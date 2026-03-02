"""Persistence helpers for writing training and evaluation metrics artifacts."""

import json
import logging
from pathlib import Path
from typing import Literal

from ml.config.validation_schemas.model_cfg import TrainModelConfig
from ml.exceptions import PersistenceError

logger = logging.getLogger(__name__)

def save_metrics(metrics: dict[str, float] | dict[str, dict[str, float]], *, model_cfg: TrainModelConfig, target_run_id: str, experiment_dir: Path, stage: Literal["training", "evaluation"]) -> str:
    """Serialize and persist metrics payload for a run stage, returning file path.

    Args:
        metrics: Metrics payload for the run stage.
        model_cfg: Validated training configuration.
        target_run_id: Target run identifier for stage-specific output.
        experiment_dir: Base experiment directory path.
        stage: Run stage name (training or evaluation).

    Returns:
        String path to the persisted metrics file.
    """

    metrics_file = experiment_dir / stage / target_run_id / "metrics.json"

    metrics_file.parent.mkdir(parents=True, exist_ok=True)

    metrics_json = {
        "task_type": model_cfg.task.type,
        "algorithm": model_cfg.algorithm.value,
        "metrics": metrics,
    }

    try:
        with open(metrics_file, "w") as f:
            json.dump(metrics_json, f, indent=2)
        logger.info(f"Metrics successfully saved to {metrics_file}.")
        return str(metrics_file)
    except Exception as e:
        msg = f"Failed to save metrics to {metrics_file}"
        logger.exception(msg)
        raise PersistenceError(msg) from e