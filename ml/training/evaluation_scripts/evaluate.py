"""Evaluation runner CLI.

This module provides a small command-line entrypoint to evaluate a trained
model defined in `configs/models.yaml`. It validates the model configuration,
dispatches to task-specific evaluators and persists evaluation results via
updater functions.

Typical usage example:
    python -m ml.training.evaluation_scripts.evaluate --name_and_version cancellation_v1

The module exposes helper functions used by the CLI and a `main()` function
which orchestrates the complete evaluation flow.
"""

# General imports
import yaml
import argparse
import logging
logger = logging.getLogger(__name__)

# Logger import
from ml.logging_config import setup_logging

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
        argparse.Namespace: Parsed CLI arguments with attribute
            `name_and_version` containing a string like "name_version".
    """

    parser = argparse.ArgumentParser(description="Evaluate a model.")

    parser.add_argument(
        "--name_and_version",
        type=str,
        required=True,
        help="Model name and version in the format 'name_version', e.g., 'cancellation_v1'"
    )

    return parser.parse_args()

def get_model_configs(name_and_version: str) -> dict:
    """Load and return model configuration for a given model identifier.

    Args:
        name_and_version (str): Model key in `configs/models.yaml` (e.g.
            "cancellation_v1").

    Returns:
        dict: The configuration dictionary for the requested model.

    Raises:
        KeyError: If the provided key is not present in the YAML file.
    """

    with open("configs/models.yaml") as f:
        configs = yaml.safe_load(f)

    try:
        return configs[name_and_version]
    except KeyError:
        raise KeyError(f"Model config '{name_and_version}' not found in models.yaml")

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

def main() -> None:
    """Orchestrate evaluation: parse args, run evaluator, persist results.

    The function executes the following high-level steps:
    1. Initialize logging.
    2. Parse CLI arguments to obtain the model identifier.
    3. Load and validate the model configuration.
    4. Run the task-specific evaluator and updater.

    This function raises on unsupported tasks or invalid configuration.
    """

    # Step 1 - Setup logging
    setup_logging()

    # Step 2 - Parse arguments
    args = parse_args()

    # Step 3 - Load model configurations
    model_configs = get_model_configs(args.name_and_version)

    # Step 4 - Validate essential keys in model configurations
    assert_keys(model_configs, ["task", "name", "version", "features", "artifacts", "threshold"])

    # Step 5 - Extract task and best threshold
    best_threshold = model_configs.get("threshold", 0.5)
    validate_threshold(best_threshold)

    task = model_configs["task"]

    # Step 6 - Define evaluator based on task
    # Evaluator registry: extend this dictionary to add more tasks
    EVALUATORS = {
        "binary_classification": evaluate_classification
    }

    key = f"{task}"
    evaluator = EVALUATORS.get(key)

    # Step 7 - Run the evaluation script
    if evaluator:
        evaluation_results = evaluator(model_configs, best_threshold)
    else:
        raise ValueError(f"Unsupported task: {task}")

    # Step 8 - Define updater based on task
    # Updater registry: extend this dictionary to add more tasks
    UPDATERS = {
        "binary_classification": update_classification_metadata
    }

    updater = UPDATERS.get(key)

    # Step 9 - Update metadata with evaluation results
    if updater:
        updater(
            model_configs,
            evaluation_results,
            best_threshold,
        )
    else:
        raise ValueError(f"Unsupported task: {task}")

if __name__ == "__main__":
    main()