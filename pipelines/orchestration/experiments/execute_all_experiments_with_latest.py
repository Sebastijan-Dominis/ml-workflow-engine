"""Bulk experiment orchestrator for all configured model specs.

Warning:
    This script is intended primarily for development usage. It delegates to a
    downstream orchestrator that defaults to "latest" snapshot resolution for
    several stages, which can produce surprising results in highly dynamic
    environments.
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

MODEL_SPECS_DIR = Path("configs/model_specs")

logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the bulk experiments runner.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description="Run all of the experiments by defaulting to latest experiment id in train, and latest experiment id, along with latest train id, in evaluate and explain.")

    parser.add_argument(
        "--env",
        choices = ["dev", "test", "prod", "default"],
        default="default",
        help="Environment to run the script in (dev/test/prod) (default: default) ~ none"
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
        help="Owner of the experiments (default: Sebastijan)"
    )

    parser.add_argument(
        "--clean-up-failure-management",
        type=str_to_bool,
        default=True,
        help="Whether to clean up the failure management folders after each experiment (default: True). Setting this to False can be useful for debugging failures, but may lead to accumulation of failure management folders over time."
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

    parser.add_argument(
        "--skip-if-existing",
        type=str_to_bool,
        default=True,
        help="Whether to skip running an experiment if at least one experiment folder exists for the model (default: True)"
    )

    return parser.parse_args()

def discover_models() -> list[tuple[str, str, str]]:
    """Discover model tuples from the model specs directory structure.

    Returns:
        list[tuple[str, str, str]]: ``(problem, segment, version)`` tuples for
        all discoverable model specs.
    """
    models: list[tuple[str, str, str]] = []
    if not MODEL_SPECS_DIR.exists():
        logger.error(f"Model specs directory does not exist: {MODEL_SPECS_DIR}")
        return models

    for problem_dir in MODEL_SPECS_DIR.iterdir():
        if not problem_dir.is_dir():
            continue
        problem = problem_dir.name
        for segment_dir in problem_dir.iterdir():
            if not segment_dir.is_dir():
                continue
            segment = segment_dir.name
            for version_file in segment_dir.glob("*.yaml"):
                version = version_file.stem
                models.append((problem, segment, version))
    return models

def run_model(
    problem: str,
    segment: str,
    version: str,
    *,
    args: argparse.Namespace,
    start_time: float
):
    """Run the end-to-end experiment orchestrator for one model tuple.

    Args:
        problem: Problem name.
        segment: Segment name.
        version: Model version.
        args: Parsed CLI arguments.
        start_time: Parent run timer start for shared completion logging.

    Returns:
        int: Return code from the delegated subprocess.
    """
    model_experiments_dir = Path("experiments") / problem / segment / version
    existing_experiment_dirs: list[Path] = [d for d in model_experiments_dir.iterdir() if d.is_dir()] if model_experiments_dir.exists() else []
    if existing_experiment_dirs and args.skip_if_existing:
        logger.info(f"Skipping model {problem}/{segment}/{version} because experiment directories already exist, and skip-if-existing is set to True. Existing experiment directories: {[d.name for d in existing_experiment_dirs]}")
        return 0
    cmd = [
        sys.executable,
        "-m", "pipelines.orchestration.experiments.execute_experiment_with_latest",
        "--problem", problem,
        "--segment", segment,
        "--version", version,
        "--env", args.env,
        "--strict", str(args.strict),
        "--logging-level", args.logging_level.upper(),
        "--owner", args.owner,
        "--clean-up-failure-management", str(args.clean_up_failure_management),
        "--overwrite-existing", str(args.overwrite_existing)
    ]
    if args.top_k is not None:
        cmd.extend(["--top-k", str(args.top_k)])

    model_start_time = time.perf_counter()
    logger.info(f"Running experiment for model: problem={problem}, segment={segment}, version={version}")

    result = subprocess.run(cmd, text=True, capture_output=True, encoding="utf-8")
    if result.returncode != 0:
        log_completion(start_time, f"Experiments run failed with model problem={problem}, segment={segment}, version={version} with return code {result.returncode}")
    else:
        log_completion(model_start_time, f"Experiment for model problem={problem}, segment={segment}, version={version} completed successfully")
    return result.returncode

def main() -> int:
    """Execute experiments for all discovered model specs.

    Returns:
        int: ``0`` on full success, otherwise non-zero.

    Notes:
        The script continues across model failures and reports aggregate status
        at the end, enabling batch execution with partial tolerance.

    Side Effects:
        Discovers model specs from disk, runs delegated experiment subprocesses,
        and writes consolidated batch logs.

    Examples:
        python -m pipelines.orchestration.experiments.execute_all_experiments_with_latest --env dev --skip-if-existing true
    """
    args = parse_args()

    start_time = time.perf_counter()

    timestamp = iso_no_colon(datetime.now())
    run_id = f"{timestamp}_{uuid4().hex[:8]}"
    log_file = Path(f"orchestration_logs/experiments/execute_all_experiments_with_latest/{run_id}/experiments_execution.log")
    log_level = getattr(logging, args.logging_level.upper(), logging.INFO)
    setup_logging(path=log_file, level=log_level)

    logger.info(f"Starting experiment execution with run id: {run_id}. Arguments: {args}")

    models = discover_models()
    if not models:
        logger.warning("No models found to run.")
        log_completion(start_time, "Experiment execution completed successfully")
        return 0

    logger.info(f"Found {len(models)} models. Starting execution...")
    return_codes = []
    for problem, segment, version in models:
        ret = run_model(
            problem,
            segment,
            version,
            args=args,
            start_time=start_time
        )
        return_codes.append(ret)
        if ret != 0:
            logger.warning(f"Continuing to next model despite failure in {problem}/{segment}/{version}")

    failed_models = [(p, s, v) for (p, s, v), code in zip(models, return_codes, strict=True) if code != 0]
    if failed_models:
        logger.warning("Failed models:")
        for problem, segment, version in failed_models:
            logger.warning(f"  - {problem}/{segment}/{version}")

    if any(code != 0 for code in return_codes):
        log_completion(start_time, "Experiment execution completed with some failures")
        return 1
    else:
        log_completion(start_time, "Experiment execution completed successfully")
        return 0

if __name__ == "__main__":
    sys.exit(main())
