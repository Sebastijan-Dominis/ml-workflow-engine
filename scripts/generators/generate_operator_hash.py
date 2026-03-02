"""Utility CLI to generate feature operator hash identifiers.

This script validates selected operator names and logs the deterministic hash
used by the feature-freezing subsystem.
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from ml.feature_freezing.utils.operators import generate_operator_hash
from ml.logging_config import setup_logging
from ml.registry.feature_operators import FEATURE_OPERATORS
from ml.utils.formatting.iso_no_col import iso_no_colon

logger = logging.getLogger(__name__)

ALLOWED = set(FEATURE_OPERATORS.keys())

def parse_args():
    """Parse command-line arguments for operator hash generation.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description="Generate operators hash for feature freezing.")
    parser.add_argument(
        "--operators", 
        nargs="+", 
        required=True, 
        choices=ALLOWED,
        help="List of feature engineering operators to include."
    )
    return parser.parse_args()

def main() -> int:
    """Generate and log a hash for the selected feature operators.

    Returns:
        int: Process exit code where ``0`` indicates success.

    Notes:
        Operator ordering affects hash output only according to downstream hash
        implementation semantics.

    Side Effects:
        Writes generation logs to a run-specific script log directory.

    Examples:
        python -m scripts.generators.generate_operator_hash --operators ArrivalDate TotalStay
    """
    timestamp = iso_no_colon(datetime.now())
    run_id = f"{timestamp}_{uuid4().hex[:8]}"
    log_file = Path(f"scripts_logs/generators/generate_operator_hash/{run_id}/op_hash_generation.log")
    log_level = logging.INFO
    setup_logging(path=log_file, level=log_level)

    args = parse_args()
    operator_names = args.operators
    
    try:
        operators_hash = generate_operator_hash(operator_names)
        logger.info(f"Generated operators hash: {operators_hash}")
        return 0
    
    except Exception as e:
        logger.error(f"Failed to generate operators hash: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
