from ml.feature_freezing.logging_config import setup_logging
import argparse

from ml.feature_freezing.utils import generate_operator_hash

def parse_args():
    parser = argparse.ArgumentParser(description="Generate operators hash for feature freezing.")

    parser.add_argument("--operators", nargs="+", required=False,
                        default=["TotalStay", "AdrPerPerson", "ArrivalSeason"],
                        help="List of feature engineering operators to include.")

    return parser.parse_args()

def main():
    setup_logging()
    args = parse_args()
    operator_names = args.operators
    operators_hash = generate_operator_hash(operator_names)
    print(f"Generated operators hash: {operators_hash}")

if __name__ == "__main__":
    main()