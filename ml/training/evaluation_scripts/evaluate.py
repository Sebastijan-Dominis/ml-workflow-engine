# General imports
import yaml
import json
import argparse

# Evaluation scripts imports
from ml.training.evaluation_scripts.custom_evaluation_scripts.evaluate_classification import (
    evaluate_classification
)

# Persistence scripts imports
from ml.training.evaluation_scripts.persistence.update_classification_metadata import (
    update_classification_metadata
)

# ---------------------------------------------------
# Helper function to parse arguments
# ---------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a model.")

    parser.add_argument(
        "--name_and_version",
        type=str,
        required=True,
        help="Model name and version in the format 'name_version', e.g., 'cancellation_v1'"
    )

    return parser.parse_args()

# ------------------------------------------
# Helper function to get model configurations
# ------------------------------------------
def get_model_configs(name_and_version):
    with open("configs/models.yaml") as f:
        configs = yaml.safe_load(f)

    return configs[name_and_version]

# ------------------------------------------
# Main evaluation script
# ------------------------------------------
def main():
    # Step 1 - Parse arguments
    args = parse_args()

    # Step 2 - Load model configurations
    model_configs = get_model_configs(args.name_and_version)

    # Step 3 - Extract task and best threshold
    best_threshold=model_configs.get("best_threshold", 0.5)
    task = model_configs["task"]

    # Step 4 - Define evaluator based on task
    EVALUATORS = {
        "binary_classification": evaluate_classification
    }

    key = f"{task}"
    evaluator = EVALUATORS.get(key)

    # Step 5 - Run the evaluation script
    if evaluator:
        evaluation_results = evaluator(model_configs, best_threshold)
    else:
        raise ValueError(f"Unsupported task: {task}")

    # Step 6 - Define updater based on task
    UPDATERS = {
        "binary_classification": update_classification_metadata
    }

    updater = UPDATERS.get(key)

    # Step 7 - Update metadata with evaluation results
    if updater:
        updater(
            model_configs,
            evaluation_results,
            best_threshold
        )
    else:
        raise ValueError(f"Unsupported task: {task}")
        
    # Step 8 - Print success message
    model_name = model_configs["name"]
    model_version = model_configs["version"]
    print(f"Evaluation and metadata update completed successfully for model '{model_name}_{model_version}'.")

if __name__ == "__main__":
    main()