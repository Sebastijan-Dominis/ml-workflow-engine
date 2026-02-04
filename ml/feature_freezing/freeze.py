import sys
import argparse
import yaml
import logging
logger = logging.getLogger(__name__)

from pathlib import Path

from ml.feature_freezing.freeze_strategies.tabular.strategy import FreezeTabular
from ml.feature_freezing.freeze_strategies.time_series import FreezeTimeSeries
from ml.feature_freezing.logging_config import setup_logging
from ml.feature_freezing.freeze_strategies.tabular.config.validate_feature_registry import validate_feature_registry
from ml.feature_freezing.persistence.save_metadata import save_metadata
from ml.cli.error_handling import resolve_exit_code
from ml.exceptions import UserError

STRATEGIES = {
    "tabular": FreezeTabular,
    "time_series": FreezeTimeSeries,
}

def parse_args():
    parser = argparse.ArgumentParser(description="Freeze features.")
    parser.add_argument("--problem", type=str, required=True, help="Problem name")
    parser.add_argument("--segment", type=str, required=True, help="Segment name")
    parser.add_argument("--feature_set", type=str, required=True, help="Feature set name")
    parser.add_argument("--version", type=str, required=True, help="Feature set version")
    return parser.parse_args()

def load_feature_registry(problem, segment, feature_set, version) -> dict:
    path = Path(f"configs/feature_registry/features.yaml")
    with open(path, "r") as f:
        registry = yaml.safe_load(f)
    return registry[problem][segment][feature_set][version]

def main() -> int:
    setup_logging()
    try:
        args = parse_args()
        config_raw = load_feature_registry(args.problem, args.segment, args.feature_set, args.version)
        config = validate_feature_registry(config_raw)

        if config.type not in STRATEGIES:
            msg = f"Unknown feature type: {config.type}"
            logger.error(msg)
            raise UserError(msg)

        strategy = STRATEGIES[config.type]()

        snapshot_path, metadata = strategy.freeze(config)
        save_metadata(snapshot_path, metadata)

        return 0

    except Exception as e:
        logger.exception("Feature freezing failed")
        return resolve_exit_code(e)

if __name__ == "__main__":
    sys.exit(main())


