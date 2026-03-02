# CRITICAL WARNING: This script is the absolute master orchestrator. It runs EVERYTHING. Use with extreme caution.

import argparse
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from ml.logging_config import setup_logging
from ml.utils.formatting.iso_no_col import iso_no_colon
from ml.utils.formatting.str_2_bol import str2bool
from ml.utils.scripts.logging import log_completion

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Absolute master orchestrator. Runs EVERYTHING."
    )

    parser.add_argument(
        "--env",
        type=str,
        default="dev",
        help="Environment to run the script in (e.g., dev, prod)"
    )

    parser.add_argument(
        "--logging-level",
        type=str,
        default="INFO"
    )

    parser.add_argument(
        "--owner",
        type=str,
        default="Sebastijan"
    )

    parser.add_argument(
        "--skip-if-existing",
        type=str2bool,
        default=True,
    )

    return parser.parse_args()


def run_step(cmd: list[str], step_name: str):
    logger.info(f"Starting step: {step_name}")
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        logger.error(f"{step_name} failed with code {result.returncode}")
        return result.returncode
    logger.info(f"Completed step: {step_name}")
    return 0


def main():
    args = parse_args()
    start_time = time.perf_counter()

    timestamp = iso_no_colon(datetime.now())
    run_id = f"{timestamp}_{uuid4().hex[:8]}"

    log_file = Path(f"scripts_logs/just_do_it/{run_id}/just_do_it.log")
    setup_logging(log_file)

    logger.info(f"JUST DO IT started. Run ID: {run_id}")

    steps = [
        (
            "Execute All Data Preprocessing",
            [
                sys.executable,
                "-m", "scripts.data.execute_all_data_preprocessing",
                "--skip-if-existing", str(args.skip_if_existing)
            ]
        ),
        (
            "Freeze All Feature Sets",
            [
                sys.executable,
                "-m", "scripts.features.freeze_all_feature_sets",
                "--logging-level", args.logging_level,
                "--owner", args.owner,
                "--skip-if-existing", str(args.skip_if_existing)
            ]
        ),
        (
            "Execute All Experiments",
            [
                sys.executable,
                "-m", "scripts.experiments.execute_all_experiments_with_latest",
                "--env", args.env,
                "--logging-level", args.logging_level,
                "--owner", args.owner,
                "--skip-if-existing", str(args.skip_if_existing)
            ]
        ),
    ]

    for step_name, cmd in steps:
        code = run_step(cmd, step_name)
        if code != 0:
            log_completion(start_time, f"JUST DO IT failed at step: {step_name}")
            return code

    log_completion(start_time, "JUST DO IT completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())