"""Command-line entrypoint for running model explainability routines.

This module provides a small CLI to load a model configuration from
`configs/models.yaml` and dispatch the appropriate explainability
implementation for the model's algorithm. Currently the module supports
CatBoost explainability via the `explain_catboost` function.

Typical usage:
    python -m ml.training.explain_scripts.explain --name_version cancellation_v1

The public functions in this module are:
    - parse_args(): parse CLI arguments
    - get_model_configs(name_version): load model configuration
    - main(): orchestrate the explainability run
"""

# General imports
import yaml
import argparse
import logging

logger = logging.getLogger(__name__)

# Explainability scripts imports
from ml.training.explain_scripts.custom_explainability_scripts.explain_catboost import explain_catboost

# Logger import
from ml.logging_config import setup_logging

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the explainability script.

    Returns:
        argparse.Namespace: Parsed arguments with attribute
            `name_version` (str) in the form "<name>_<version>".
    """

    parser = argparse.ArgumentParser(description="Explain a model.")

    parser.add_argument(
        "--name_version",
        type=str,
        required=True,
        help="Model name and version in the format 'name_version', e.g., 'cancellation_v1'",
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

def main() -> None:
    """Orchestrate the explainability run for a model.

    The function performs the following high-level steps:
        1. Set up logging using `setup_logging()`.
        2. Parse CLI arguments to obtain `name_version`.
        3. Load model configuration from `configs/models.yaml`.
        4. Select and invoke the appropriate explainer based on the
           configuration's `algorithm` field.

    Raises:
        ValueError: If the model's algorithm is not supported by this script.
        KeyError: If the requested model configuration is missing required
            fields or cannot be found.
    """

    # Step 1 - Setup logging
    setup_logging()

    # Step 2 - Parse arguments
    args = parse_args()

    # Step 3 - Load model configurations
    model_configs = get_model_configs(args.name_version)

    # Step 4 - Extract algorithm
    algorithm = model_configs["algorithm"]

    # Step 5 - Define explainer based on algorithm
    # Explainer directory - update as new explainers are added
    EXPLAINERS = {
        "catboost": explain_catboost
    }

    key = f"{algorithm}"
    explainer = EXPLAINERS.get(key)

    # Step 6 - Run the explainability script
    if explainer:
        explainer(model_configs)
    else:
        msg = f"Unsupported algorithm: {algorithm}"
        logger.error(msg)
        raise ValueError(msg)

if __name__ == "__main__":
    main()