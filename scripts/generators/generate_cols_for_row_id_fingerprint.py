"""Utility script to generate row-id column fingerprint values.

The script computes and logs a deterministic fingerprint for the configured
``cols_for_row_id`` sequence used in processed data row identifier logic.
"""

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
from ml.utils.formatting.iso_no_col import iso_no_colon

logger = logging.getLogger(__name__)

def main() -> int:
    """Compute and log fingerprint for row-id column definitions.

    Returns:
        int: Process exit code where ``0`` indicates success.

    Notes:
        Fingerprint stability depends on the canonical row-id column definition
        source used by processed data generation.

    Side Effects:
        Writes generation logs to a run-specific script log directory.

    Examples:
        python -m scripts.generators.generate_cols_for_row_id_fingerprint
    """
    timestamp = iso_no_colon(datetime.now())
    run_id = f"{timestamp}_{uuid4().hex[:8]}"
    log_file = Path(f"scripts_logs/generators/generate_cols_for_row_id_fingerprint/{run_id}/cols_fingerprint.log")
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