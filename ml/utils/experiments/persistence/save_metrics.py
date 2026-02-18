import json
import logging
from pathlib import Path
from typing import Literal

from ml.config.validation_schemas.model_cfg import TrainModelConfig
from ml.exceptions import PersistenceError

logger = logging.getLogger(__name__)

def save_metrics(metrics: dict[str, float] | dict[str, dict[str, float]], *, model_cfg: TrainModelConfig, target_run_id: str, experiment_dir: Path, stage: Literal["training", "evaluation"]) -> str:
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