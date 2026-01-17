# General imports
import yaml
import argparse

# Explainability scripts imports
from ml.training.explain_scripts.custom_explainability_scripts.explain_catboost import explain_catboost

# ---------------------------------------------------
# Helper function to parse arguments
# ---------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Explain a model.")

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

    try:
        return configs[name_and_version]
    except KeyError:
        raise KeyError(f"Model config '{name_and_version}' not found in models.yaml")


# ------------------------------------------
# Main function for explaining a model
# ------------------------------------------
def main():
    # Step 1 - Parse arguments
    args = parse_args()

    # Step 2 - Load model configurations
    model_configs = get_model_configs(args.name_and_version)

    # Step 3 - Extract algorithm
    algorithm = model_configs["algorithm"]

    # Step 4 - Define explainer based on algorithm
    EXPLAINERS = {
        "catboost": explain_catboost
    }

    key = f"{algorithm}"
    explainer = EXPLAINERS.get(key)

    # Step 5 - Run the explainability script
    if explainer:
        explainer(model_configs)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    # Step 6 - Print success message
    print(f"Explainability for model {args.name_and_version} completed successfully.")

if __name__ == "__main__":
    main()