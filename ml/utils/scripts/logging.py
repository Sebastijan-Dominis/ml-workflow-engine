"""Script logging helpers for standardized completion messages."""

import logging
import time
from datetime import datetime

from ml.utils.formatting.iso_no_colon import iso_no_colon

logger = logging.getLogger(__name__)


def log_completion(start_time: float, message: str):
    """Log operation completion timestamp and human-readable duration.

    Args:
        start_time: Monotonic start time captured before operation execution.
        message: Completion message prefix.

    Returns:
        None.
    """

    end_time = time.perf_counter()
    duration = end_time - start_time
    # Decide whether to use seconds, minutes, or hours based on duration
    if duration < 60:
        duration_str = f"{duration:.2f} seconds"
    elif duration < 3600:
        duration_str = f"{duration/60:.2f} minutes"
    else:
        duration_str = f"{duration/3600:.2f} hours"
    end = iso_no_colon(datetime.now())
    logger.info(f"{message} at {end} after {duration_str}.")