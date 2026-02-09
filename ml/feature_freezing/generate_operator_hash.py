import logging
import argparse
import sys
from pathlib import Path

from ml.cli.error_handling import resolve_exit_code
from ml.logging_config import setup_logging
from ml.feature_freezing.utils.operators import generate_operator_hash

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate operators hash for feature freezing.")
    parser.add_argument("--operators", nargs="+", required=False,
                        default=["TotalStay", "AdrPerPerson", "ArrivalSeason"],
                        help="List of feature engineering operators to include.")
    return parser.parse_args()

def main() -> int:
    path = Path("ml/feature_freezing/logs/generate_operator_hash.log")
    path.parent.mkdir(parents=True, exist_ok=True)
    setup_logging(path)
    try:
        args = parse_args()
        operator_names = args.operators
        operators_hash = generate_operator_hash(operator_names)
        print(f"Generated operators hash: {operators_hash}")
        return 0
    except Exception as e:
        logger.exception("Failed to generate operators hash")
        return resolve_exit_code(e)

if __name__ == "__main__":
    import logging
    import sys
    sys.exit(main())

