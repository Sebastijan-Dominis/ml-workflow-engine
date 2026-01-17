# General imports
import yaml
import argparse

# Specific training script imports
from ml.training.train_scripts.custom_training_scripts.binary_classification_catboost import (
    train_binary_classification_with_catboost
)

# Persistence imports
from ml.training.train_scripts.persistence.save_pipeline_and_metadata import (
    save_pipeline_and_metadata
)

from ml.training.train_scripts.persistence.update_general_config import (
    update_general_config
)

# ---------------------------------------------------
# Helper function to parse arguments
# ---------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train a model.")

    parser.add_argument(
        "--name_and_version",
        type=str,
        required=True,
        help="Model name and version in the format 'name_version', e.g., 'cancellation_v1'"
    )

    return parser.parse_args()

# ------------------------------------------
# Helper function to load config
# ------------------------------------------
def load_config(name_and_version):   
    with open(f"ml/training/train_configs/{name_and_version}.yaml") as f:
        cfg = yaml.safe_load(f)

    return cfg

# ---------------------------------------------------
# Main function to run the training script
# ---------------------------------------------------
def main():
    # Step 1 - Parse arguments
    args = parse_args()

    # Step 2 - Load config
    cfg = load_config(args.name_and_version)

    # Step 3 - Extract task and algorithm
    task = cfg["task"]
    algorithm = cfg["model"]["algorithm"]

    # Step 4 - Define trainer based on task and algorithm
    TRAINERS = {
        "binary_classification_catboost": train_binary_classification_with_catboost
    }

    key = f"{task}_{algorithm}"
    trainer = TRAINERS.get(key)

    # Step 5 - Run the training script
    if trainer:
        pipeline = trainer(args.name_and_version, cfg)
    else:
        raise ValueError(f"Unsupported task and algorithm: {task}_{algorithm}")
    
    # Step 6 - Save pipeline + metadata
    save_pipeline_and_metadata(pipeline, cfg)

    # Step 7 - Update general model config file
    update_general_config(cfg)

if __name__ == "__main__":
    main()