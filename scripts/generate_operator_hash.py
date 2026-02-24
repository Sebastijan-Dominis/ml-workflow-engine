import argparse

from pathlib import Path
from ml.feature_freezing.utils.operators import generate_operator_hash

from ml.registry.feature_operators import FEATURE_OPERATORS

ALLOWED = set(FEATURE_OPERATORS.keys())

def parse_args():
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
    path = Path("ml/feature_freezing/logs/generate_operator_hash.log")
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        args = parse_args()
        operator_names = args.operators
        operators_hash = generate_operator_hash(operator_names)
        print(f"Generated operators hash: {operators_hash}")
        return 0
    
    except Exception as e:
        print(f"Failed to generate operators hash: {e}")
        return 1

if __name__ == "__main__":
    main()

