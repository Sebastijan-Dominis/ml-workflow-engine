import logging
from typing import Literal

from ml.exceptions import DataError

logger = logging.getLogger(__name__)

def compute_memory_change(*, target_metadata: dict, new_memory_usage: float, stage: Literal["interim", "processed"]) -> dict:
    try:
        if stage == "interim":
            old_memory_usage = target_metadata["memory_usage_mb"]
        elif stage == "processed":
            old_memory_usage = target_metadata["memory"]["new_memory_mb"]
        change = new_memory_usage - old_memory_usage
        return {
            "old_memory_mb": old_memory_usage,
            "new_memory_mb": new_memory_usage,
            "change_mb": change,
            "change_percentage": (change / old_memory_usage * 100) if old_memory_usage > 0 else 0
        }
    except KeyError as e:
        msg = f"Target metadata is missing the key required to compute memory improvement."
        logger.error(msg)
        raise DataError(msg) from e