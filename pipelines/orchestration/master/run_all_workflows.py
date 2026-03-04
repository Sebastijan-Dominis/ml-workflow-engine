"""Master orchestration script that executes all major project workflows.

Warning:
    This script triggers full preprocessing, feature freezing, and experiment
    execution. It is intentionally powerful and should be run with care.
"""

import argparse
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from ml.io.formatting.iso_no_colon import iso_no_colon
from ml.io.formatting.str_to_bool import str_to_bool
from ml.logging_config import setup_logging
from pipelines.orchestration.common.orchestration_logging import log_completion

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the master orchestrator.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="Absolute master orchestrator. Runs full project orchestration pipeline."
    )

    parser.add_argument(
        "--env",
        choices=["dev", "test", "prod", "default"],
        default="dev",
        help="Environment to run the script in (e.g., dev, prod)"
    )

    parser.add_argument(
        "--logging-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO"
    )

    parser.add_argument(
        "--owner",
        type=str,
        default="Sebastijan"
    )

    parser.add_argument(
        "--skip-if-existing",
        type=str_to_bool,
        default=True,
    )

    return parser.parse_args()


def run_step(cmd: list[str], step_name: str) -> int:
    """Run one orchestration step command with structured logging.

    Args:
        cmd: Command and arguments for the step.
        step_name: Human-readable step name for logs.

    Returns:
        int: Subprocess return code (``0`` on success).
    """
    logger.info(f"Starting step: {step_name}")
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        logger.error(f"{step_name} failed with code {result.returncode}")
        return result.returncode
    logger.info(f"Completed step: {step_name}")
    return 0


def main() -> int:
    """Execute all high-level orchestration steps in sequence.

    Returns:
        int: Process exit code where ``0`` indicates success.

    Notes:
        This is a high-impact orchestrator that chains full preprocessing,
        freezing, and experiment execution; partial failures stop later stages.

    Side Effects:
        Launches multiple long-running subprocess workflows and writes a single
        master orchestration log.

    Examples:
        python -m pipelines.orchestration.master.run_all_workflows --env dev --skip-if-existing true
    """
    args = parse_args()
    start_time = time.perf_counter()

    timestamp = iso_no_colon(datetime.now())
    run_id = f"{timestamp}_{uuid4().hex[:8]}"

    log_file = Path(f"orchestration_logs/run_all_workflows/{run_id}/run_all_workflows.log")
    setup_logging(log_file)

    logger.info(f"Run all workflows started. Run ID: {run_id}")

    steps = [
        (
            "Execute All Data Preprocessing",
            [
                sys.executable,
                "-m", "pipelines.orchestration.data.execute_all_data_preprocessing",
                "--skip-if-existing", str(args.skip_if_existing)
            ]
        ),
        (
            "Freeze All Feature Sets",
            [
                sys.executable,
                "-m", "pipelines.orchestration.features.freeze_all_feature_sets",
                "--logging-level", args.logging_level,
                "--owner", args.owner,
                "--skip-if-existing", str(args.skip_if_existing)
            ]
        ),
        (
            "Execute All Experiments",
            [
                sys.executable,
                "-m", "pipelines.orchestration.experiments.execute_all_experiments_with_latest",
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
            log_completion(start_time, f"Run all workflows failed at step: {step_name}")
            return code

    log_completion(start_time, "Run all workflows completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
