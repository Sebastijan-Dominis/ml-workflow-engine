"""Persistence helpers for writing training and evaluation metrics artifacts."""

import json
import logging
from pathlib import Path
from typing import Literal

from ml.config.schemas.model_cfg import TrainModelConfig
from ml.exceptions import PersistenceError
from ml.modeling.models.metrics import EvaluationMetrics, TrainingMetrics
from ml.modeling.validation.metrics import validate_evaluation_metrics, validate_training_metrics

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
        validated_metrics: TrainingMetrics | EvaluationMetrics
        if stage == "training":
            validated_metrics = validate_training_metrics(metrics_json)
        elif stage == "evaluation":
            validated_metrics = validate_evaluation_metrics(metrics_json)
        with open(metrics_file, "w") as f:
            json.dump(validated_metrics.model_dump(exclude_none=True), f, indent=2)
        logger.info(f"Metrics successfully saved to {metrics_file}.")
        return str(metrics_file)
    except Exception as e:
        msg = f"Failed to save metrics to {metrics_file}"
        logger.exception(msg)
        raise PersistenceError(msg) from e
