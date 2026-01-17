import json

from pathlib import Path

# ------------------------------------------
# Helper function to get metadata file path
# ------------------------------------------
def get_file(model_configs):
    metadata_file = Path(model_configs["artifacts"]["metadata"])
    return metadata_file

# ------------------------------------------
# Helper function to load metadata
# ------------------------------------------
def load_metadata(metadata_file):
    with open(metadata_file, "r") as f:
        metadata = json.load(f)
    return metadata

# ------------------------------------------
# Helper function to update metadata content with evaluation results
# ------------------------------------------
def update_content(metadata, evaluation_results, best_threshold):
    metadata.setdefault("metrics", {})
    for split_name, metrics in evaluation_results.items():
        metadata["metrics"][split_name] = metrics
    metadata["threshold"] = best_threshold

# ------------------------------------------
# Helper function to save metadata
# ------------------------------------------
def save_metadata(metadata, metadata_file):
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

# ------------------------------------------
# Main function to update metadata with evaluation results
# ------------------------------------------
def update_classification_metadata(model_configs, evaluation_results, best_threshold):
    # Step 1 - get metadata file path
    metadata_file = get_file(model_configs)

    # Step 2 - load existing metadata
    metadata = load_metadata(metadata_file)

    # Step 3 - update metadata content
    update_content(metadata, evaluation_results, best_threshold)

    # Step 4 - save updated metadata
    save_metadata(metadata, metadata_file)

    # Step 5 - print success message
    print("Metadata updated successfully.")