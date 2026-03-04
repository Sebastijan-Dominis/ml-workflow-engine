"""Validation helpers for threshold compatibility and value constraints."""

import logging
from pathlib import Path

from ml.config.schemas.model_specs import TaskConfig
from ml.exceptions import ConfigError
from ml.policies.promotion.threshold_support import TASKS_SUPPORTING_THRESHOLDS
from ml.utils.loaders import load_json

logger = logging.getLogger(__name__)

def validate_threshold(task: TaskConfig, metrics_path: Path) -> float | None:
    """Return validated threshold for supported tasks or `None` when unsupported.

    Args:
        task: Task configuration describing problem type/subtype.
        metrics_path: Metrics file path potentially containing a threshold value.

    Returns:
        Validated threshold value, default threshold, or ``None`` when unsupported.
    """

    key = (task.type.lower(), task.subtype.lower() if task.subtype else None)
    if key not in TASKS_SUPPORTING_THRESHOLDS:
        logger.debug(f"Task type '{task.type}' does not support thresholds. Skipping threshold validation.")
        return None

    metrics = load_json(metrics_path)
    threshold = metrics.get("metrics", {}).get("threshold", {}).get("value")

    if threshold is None:
        logger.warning("No threshold value found in metrics. Defaulting to 0.5.")
        return 0.5

    if threshold > 1.0 or threshold < 0.0:
        msg = f"Invalid threshold value: {threshold}. It must be between 0 and 1."
        logger.error(msg)
        raise ConfigError(msg)

    logger.debug(f"Threshold value {threshold} is valid.")

    return threshold
