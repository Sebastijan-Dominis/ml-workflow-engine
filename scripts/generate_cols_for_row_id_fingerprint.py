import logging
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from ml.data.processed.processing.hotel_bookings.cols_for_row_id import \
    cols_for_row_id
from ml.logging_config import setup_logging
from ml.utils.data.compute_cols_for_row_id_fingerprint import \
    compute_cols_for_row_id_fingerprint
from ml.utils.iso_no_col import iso_no_colon

logger = logging.getLogger(__name__)

def main() -> int:
    timestamp = iso_no_colon(datetime.now())
    run_id = f"{timestamp}_{uuid4().hex[:8]}"
    log_file = Path(f"scripts_logs/generate_cols_for_row_id_fingerprint/{run_id}/cols_fingerprint.log")
    log_level = logging.INFO
    setup_logging(path=log_file, level=log_level)

    try:
        fingerprint = compute_cols_for_row_id_fingerprint(cols_for_row_id)
        logger.info(f"Fingerprint for cols_for_row_id: {fingerprint}")
        return 0
    except Exception as e:
        logger.error(f"Failed to compute fingerprint: {e}")
        return 1
    
if __name__ == "__main__":
    sys.exit(main())