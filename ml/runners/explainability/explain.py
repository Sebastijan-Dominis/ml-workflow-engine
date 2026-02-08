"""Command-line entrypoint for running model explainability routines.

This module provides a small CLI to load a model configuration from
`configs/models.yaml` and dispatch the appropriate explainability
implementation for the model's algorithm. Currently the module supports
CatBoost explainability via the `explain_catboost` function.

Typical usage:
    python -m ml.training.explain_scripts.explain \\
        --problem cancellation --segment global --version v1 \\
        --experiment-id 20260206_154343_2f5c2000

The public functions in this module are:
    - parse_args(): parse CLI arguments
    - get_model_configs(name_version): load model configuration
    - main(): orchestrate the explainability run
"""

# General imports
import sys
import yaml
import argparse
import logging

logger = logging.getLogger(__name__)
from pathlib import Path

# Explainability scripts imports
from ml.runners.explainability.custom_explainability_scripts.explain_catboost import explain_catboost

# Logger import
from ml.logging_config import setup_logging
from ml.cli.error_handling import resolve_exit_code

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the explainability script.

    Returns:
        argparse.Namespace: Parsed arguments.
    """

    parser = argparse.ArgumentParser(description="Explain a model.")

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
    """Load model configuration for a named model/version.

    Args:
        name_version (str): Model identifier in the form "<name>_<version>".

    Returns:
        dict: Configuration dictionary for the requested model as defined in
            `configs/models.yaml`.

    Raises:
        KeyError: If `name_version` is not present in the YAML file.
        FileNotFoundError: If `configs/models.yaml` cannot be opened.
    """

    with open("configs/models.yaml") as f:
        configs = yaml.safe_load(f)

    try:
        return configs[name_version]
    except KeyError:
        logger.exception(f"Model config '{name_version}' not found in models.yaml")
        raise

def main() -> int:
    """Orchestrate the explainability run for a model.

    The function performs the following high-level steps:
        1. Set up logging inside the experiment directory.
        2. Parse CLI arguments.
        3. Load model configuration from `configs/models.yaml`.
        4. Select and invoke the appropriate explainer based on the
           configuration's `algorithm` field.

    Returns:
        0 on success, non-zero exit code on failure.
    """

    # Step 1 - Parse arguments
    args = parse_args()

    # Step 2 - Setup logging inside the experiment directory
    experiment_dir = Path("experiments") / args.problem / args.segment / args.version / args.experiment_id
    log_level = getattr(logging, args.logging_level.upper(), logging.INFO)
    setup_logging(experiment_dir / "explain.log", level=log_level)

    try:
        # Step 3 - Build name_version key for backward-compatible config lookup
        name_version = f"{args.problem}_{args.segment}_{args.version}"

        # Step 4 - Load model configurations
        model_configs = get_model_configs(name_version)

        # Step 5 - Extract algorithm
        algorithm = model_configs["algorithm"]

        # Step 6 - Define explainer based on algorithm
        EXPLAINERS = {
            "catboost": explain_catboost
        }

        key = f"{algorithm}"
        explainer = EXPLAINERS.get(key)

        # Step 7 - Run the explainability script
        if explainer:
            explainer(model_configs)
        else:
            msg = f"Unsupported algorithm: {algorithm}"
            logger.error(msg)
            raise ValueError(msg)

        logger.info(
            "Explainability completed | problem=%s segment=%s version=%s experiment_id=%s",
            args.problem,
            args.segment,
            args.version,
            args.experiment_id,
        )

        return 0

    except Exception as e:
        logger.exception("An error occurred during explainability.")
        return resolve_exit_code(e)

if __name__ == "__main__":
    sys.exit(main())