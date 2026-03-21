"""Single-model experiment orchestrator using staged pipeline CLIs.

Warning:
    This script intentionally relies on "latest" snapshot resolution for
    selected stages unless explicit identifiers are provided by invoked
    downstream commands. Review logs carefully in concurrent environments.
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
    """Parse command-line arguments for single-model orchestration.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description="Run the entire experiment by defaulting to latest experiment id in train, and latest experiment id, along with latest train id, in evaluate and explain.")

    parser.add_argument(
        "--problem",
        type=str,
        required=True,
        help="Model problem, e.g., 'no_show'"
    )

    parser.add_argument(
        "--segment",
        type=str,
        required=True,
        help="Model segment name, e.g., 'city_hotel_online_ta'"
    )

    parser.add_argument(
        "--version",
        type=str,
        required=True,
        help="Model version, e.g., 'v1'"
    )

    parser.add_argument(
        "--env",
        choices=["dev", "test", "prod", "default"],
        default="dev",
        help="Environment to run the script in (dev/test/prod) (default: dev)"
    )

    parser.add_argument(
        "--strict",
        type=str_to_bool,
        default=True,
        help="Whether to run in strict mode, which includes strict validation that may be computationally expensive (default: True)"
    )

    parser.add_argument(
        "--logging-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) (default: INFO)"
    )

    parser.add_argument(
        "--owner",
        type=str,
        default="Sebastijan",
        help="Owner of the experiment (default: Sebastijan)"
    )

    parser.add_argument(
        "--clean-up-failure-management",
        type=str_to_bool,
        default=True,
        help="Whether to clean up failure management folder after successful run (default: True)"
    )

    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="Experiment ID to use for this run (default: None, which generates a new unique experiment ID). If provided, it should be in the format 'timestamp_randomstring', e.g., '20240101T120000_abcdef12'."
    )

    parser.add_argument(
        "--overwrite-existing",
        type=str_to_bool,
        default=False,
        help="Whether to overwrite existing metadata and runtime snapshot files if they already exist in the target directory (default: False). If set to False and such files already exist, the script will raise an error to prevent accidental overwriting. Set to True to allow overwriting existing files."
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of top features to include in the explainability output (will programmatically default to settings-specific values if not provided, but can be overridden with this flag)"
    )

    return parser.parse_args()

def main() -> int:
    """Run search, train, evaluate, and explain steps for one model config.

    Returns:
        int: Process exit code where ``0`` indicates success.

    Notes:
        Orchestration relies on downstream commands that may resolve latest run
        identifiers dynamically; concurrent writes can influence selection.

    Side Effects:
        Invokes search/train/evaluate/explain subprocesses and writes aggregated
        experiment orchestration logs.

    Examples:
        python -m pipelines.orchestration.experiments.execute_experiment_with_latest --problem no_show --segment global --version v1
    """
    args = parse_args()

    start_time = time.perf_counter()

    timestamp = iso_no_colon(datetime.now())
    run_id = f"{timestamp}_{uuid4().hex[:8]}"
    log_file = Path(f"orchestration_logs/experiments/execute_experiment_with_latest/{run_id}/experiment_execution.log")
    log_level = getattr(logging, args.logging_level.upper(), logging.INFO)
    setup_logging(path=log_file, level=log_level)

    logger.info(f"Starting experiment execution with run id: {run_id}. Arguments: {args}")

    try:
        # Run search.py
        search_cmd = [
            sys.executable,
            "-m", "pipelines.search.search",
            "--problem", args.problem,
            "--segment", args.segment,
            "--version", args.version,
            "--env", args.env,
            "--strict", str(args.strict),
            "--logging-level", args.logging_level,
            "--owner", args.owner,
            "--clean-up-failure-management", str(args.clean_up_failure_management),
            "--overwrite-existing", str(args.overwrite_existing)
        ]
        if args.experiment_id:
            search_cmd.extend(["--experiment-id", args.experiment_id])
        logger.info(f"Running hyperparameter search with command: {' '.join(search_cmd)}")
        subprocess.run(search_cmd, check=True)

        # Run train.py
        train_cmd = [
            sys.executable,
            "-m", "pipelines.runners.train",
            "--problem", args.problem,
            "--segment", args.segment,
            "--version", args.version,
            "--env", args.env,
            "--strict", str(args.strict),
            "--logging-level", args.logging_level
        ]
        logger.info(f"Running training with command: {' '.join(train_cmd)}")
        train_result = subprocess.run(train_cmd, check=True, capture_output=True, text=True, encoding="utf-8")
        logger.info(f"Training output:\n{train_result.stdout}")

        # Run evaluate.py
        evaluate_cmd = [
            sys.executable,
            "-m", "pipelines.runners.evaluate",
            "--problem", args.problem,
            "--segment", args.segment,
            "--version", args.version,
            "--env", args.env,
            "--strict", str(args.strict),
            "--logging-level", args.logging_level
        ]
        logger.info(f"Running evaluation with command: {' '.join(evaluate_cmd)}")
        evaluate_result = subprocess.run(evaluate_cmd, check=True, capture_output=True, text=True, encoding="utf-8")
        logger.info(f"Evaluation output:\n{evaluate_result.stdout}")

        # Run explain.py
        explain_cmd = [
            sys.executable,
            "-m", "pipelines.runners.explain",
            "--problem", args.problem,
            "--segment", args.segment,
            "--version", args.version,
            "--env", args.env,
            "--strict", str(args.strict),
            "--logging-level", args.logging_level,
        ]
        if args.top_k is not None:
            explain_cmd.extend(["--top-k", str(args.top_k)])

        logger.info(f"Running explainability with command: {' '.join(explain_cmd)}")
        explain_result = subprocess.run(explain_cmd, check=True, capture_output=True, text=True, encoding="utf-8")
        logger.info(f"Explainability output:\n{explain_result.stdout}")

        log_completion(start_time, f"Experiment execution completed successfully for problem={args.problem}, segment={args.segment}, version={args.version}")
        return 0
    except subprocess.CalledProcessError as e:
        first_line = e.stderr.splitlines()[0] if e.stderr else ""
        logger.error(f"Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}. Error: {first_line}")
        log_completion(start_time, f"Experiment execution failed with return code {e.returncode}")
        return e.returncode

if __name__ == "__main__":
    sys.exit(main())
