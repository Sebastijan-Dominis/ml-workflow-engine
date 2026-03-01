import logging
import time
from datetime import datetime
from pathlib import Path

from ml.utils.iso_no_col import iso_no_colon

logger = logging.getLogger(__name__)


def log_completion(start_time: float, message: str):
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