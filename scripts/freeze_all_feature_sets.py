import argparse
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from ml.logging_config import setup_logging
from ml.utils.iso_no_col import iso_no_colon
from ml.utils.loaders import load_yaml

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Freeze features.")

    parser.add_argument(
        "--logging-level", 
        type=str, 
        default="INFO", 
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) (default: INFO)"
    )

    parser.add_argument(
        "--owner",
        type=str,
        default="Sebastijan",
        help="Owner of the feature sets (default: Sebastijan)"
    )

    return parser.parse_args()

def log_completion(start_time: float, message: str):
    end_time = time.perf_counter()
    duration = end_time - start_time
    end = iso_no_colon(datetime.now())
    logger.info(f"{message} at {end} after {duration:.2f} seconds")

def main() -> int:
    args = parse_args()

    start_time = time.perf_counter()

    timestamp = iso_no_colon(datetime.now())
    run_id = f"{timestamp}_{uuid4().hex[:8]}"
    log_file = Path(f"scripts_logs/freeze_all_feature_sets/{run_id}/freeze_all.log")
    log_level = getattr(logging, args.logging_level.upper(), logging.INFO)
    setup_logging(path=log_file, level=log_level)

    logger.info(f"Script started at {timestamp} with run ID {run_id}, logging level {args.logging_level.upper()} and owner {args.owner}")

    successes_count = 0

    try:
        feature_registry = load_yaml(Path("configs/feature_registry/features.yaml"))
    except Exception as e:
        logger.exception("Failed to load feature registry.")
        return 1
    
    for feature_set_name in feature_registry.keys():
        for feature_set_version in feature_registry[feature_set_name].keys():
            logger.info(f"Freezing feature set '{feature_set_name}' version '{feature_set_version}'...")
            
            cmd = [
                sys.executable,
                "-m", "pipelines.features.freeze",
                "--feature-set", feature_set_name,
                "--version", feature_set_version,
                "--logging-level", args.logging_level.upper(),
                "--owner", args.owner
            ]

            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True, encoding="utf-8")
                logger.info(f"Feature set '{feature_set_name}' succeeded.")
                successes_count += 1
            except subprocess.CalledProcessError as e:
                first_line = e.stderr.splitlines()[0] if e.stderr else ""
                logger.error(f"Failed to freeze '{feature_set_name}' version '{feature_set_version}': {first_line}")
                log_completion(start_time, f"Script terminated after successfully freezing {successes_count} feature sets")
                return e.returncode

    log_completion(start_time, f"Script completed successfully after freezing {successes_count} feature sets ")
    return 0
    
if __name__ == "__main__":
    sys.exit(main())