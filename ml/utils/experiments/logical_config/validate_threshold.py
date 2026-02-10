import logging
from pathlib import Path

from ml.config.validation_schemas.model_specs import TaskConfig
from ml.utils.loaders import load_json
from ml.exceptions import ConfigError
from ml.registry.tasks_supporting_thresholds import TASKS_SUPPORTING_THRESHOLDS

logger = logging.getLogger(__name__)

def validate_threshold(task: TaskConfig, metrics_path: Path) -> float | None:
    key = (task.type.lower(), task.subtype.lower() if task.subtype else None)
    if key not in TASKS_SUPPORTING_THRESHOLDS:
        logger.debug(f"Task type '{task.type}' does not support thresholds. Skipping threshold validation.")
        return None
    
    metrics = load_json(metrics_path)
    threshold = metrics.get("threshold", {}).get("value")

    if threshold is None:
        logger.warning("No threshold value found in metrics. Defaulting to 0.5.")
        return 0.5

    if threshold > 1.0 or threshold < 0.0:
        msg = f"Invalid threshold value: {threshold}. It must be between 0 and 1."
        logger.error(msg)
        raise ConfigError(msg)

    logger.debug(f"Threshold value {threshold} is valid.")
    
    return threshold