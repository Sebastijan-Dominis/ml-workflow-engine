"""Evaluation runner CLI.

This module provides a small command-line entrypoint to evaluate a trained
model. It validates the model configuration, dispatches to task-specific
evaluators and persists evaluation results via updater functions.

Typical usage example:
    python -m ml.training.evaluation_scripts.evaluate \\
        --problem cancellation --segment global --version v1 \\
        --experiment-id 20260206_154343_2f5c2000

The module exposes helper functions used by the CLI and a `main()` function
which orchestrates the complete evaluation flow.
"""

# General imports
import sys
import yaml
import argparse
import logging
logger = logging.getLogger(__name__)
from pathlib import Path

# Logger import
from ml.logging_config import setup_logging
from ml.cli.error_handling import resolve_exit_code

# Evaluation scripts imports
from ml.training.evaluation_scripts.custom_evaluation_scripts.evaluate_classification import (
    evaluate_classification
)

# Persistence scripts imports
from ml.training.evaluation_scripts.persistence.update_classification_metadata import (
    update_classification_metadata
)

# Utility imports
from ml.training.evaluation_scripts.utils import (
    assert_keys
)

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """

    parser = argparse.ArgumentParser(description="Evaluate a model.")

    parser.add_argument(
        "--problem",
        type=str,
        required=True,
        help="Model problem, e.g., 'cancellation'"
    )

    parser.add_argument(
        "--segment",
        type=str,
        required=True,
        help="Model segment name, e.g., 'global'"
    )

    parser.add_argument(
        "--version",
        type=str,
        required=True,
        help="Model version, e.g., 'v1'"
    )

    parser.add_argument(
        "--experiment-id",
        type=str,
        required=True,
        help="Experiment id (directory name under experiments/{problem}/{segment}/{version})"
    )

    parser.add_argument(
        "--logging-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) (default: INFO)"
    )

    return parser.parse_args()

def get_model_configs(name_version: str) -> dict:
    """Load and return model configuration for a given model identifier.

    Args:
        name_version (str): Model key in `configs/models.yaml` (e.g.
            "cancellation_v1").

    Returns:
        dict: The configuration dictionary for the requested model.

    Raises:
        KeyError: If the provided key is not present in the YAML file.
    """

    with open("configs/models.yaml") as f:
        configs = yaml.safe_load(f)

    try:
        return configs[name_version]
    except KeyError:
        raise KeyError(f"Model config '{name_version}' not found in models.yaml")

def validate_threshold(threshold: float) -> None:
    """Validate a probability threshold is within [0.0, 1.0].

    Args:
        threshold (float): Probability threshold to validate.

    Raises:
        ValueError: If `threshold` is outside the inclusive range [0.0, 1.0].
    """

    if threshold > 1.0 or threshold < 0.0:
        logger.error(f"Invalid threshold value: {threshold}. It must be between 0 and 1.")
        raise ValueError(f"Invalid threshold value: {threshold}. It must be between 0 and 1.")

def main() -> int:
    """Orchestrate evaluation: parse args, run evaluator, persist results.

    The function executes the following high-level steps:
    1. Initialize logging inside the experiment directory.
    2. Parse CLI arguments to obtain the model identifier.
    3. Load and validate the model configuration.
    4. Run the task-specific evaluator and updater.

    Returns:
        0 on success, non-zero exit code on failure.
    """

    # Step 1 - Parse arguments
    args = parse_args()

    # Step 2 - Setup logging inside the experiment directory
    experiment_dir = Path("experiments") / args.problem / args.segment / args.version / args.experiment_id
    log_level = getattr(logging, args.logging_level.upper(), logging.INFO)
    setup_logging(experiment_dir / "evaluate.log", level=log_level)

    try:
        # Step 3 - Build name_version key for backward-compatible config lookup
        name_version = f"{args.problem}_{args.segment}_{args.version}"

        # Step 4 - Load model configurations
        model_configs = get_model_configs(name_version)

        # Step 5 - Validate essential keys in model configurations
        assert_keys(model_configs, ["task", "name", "version", "features", "artifacts", "threshold"])

        # Step 6 - Extract task and best threshold
        best_threshold = model_configs.get("threshold", 0.5)
        validate_threshold(best_threshold)

        task = model_configs["task"]

        # Step 7 - Define evaluator based on task
        EVALUATORS = {
            "binary_classification": evaluate_classification
        }

        key = f"{task}"
        evaluator = EVALUATORS.get(key)

        # Step 8 - Run the evaluation script
        if evaluator:
            evaluation_results = evaluator(model_configs, best_threshold)
        else:
            raise ValueError(f"Unsupported task: {task}")

        # Step 9 - Define updater based on task
        UPDATERS = {
            "binary_classification": update_classification_metadata
        }

        updater = UPDATERS.get(key)

        # Step 10 - Update metadata with evaluation results
        if updater:
            updater(
                model_configs,
                evaluation_results,
                best_threshold,
            )
        else:
            raise ValueError(f"Unsupported task: {task}")

        logger.info(
            "Evaluation completed | problem=%s segment=%s version=%s experiment_id=%s",
            args.problem,
            args.segment,
            args.version,
            args.experiment_id,
        )

        return 0

    except Exception as e:
        logger.exception("An error occurred during evaluation.")
        return resolve_exit_code(e)

if __name__ == "__main__":
    sys.exit(main())