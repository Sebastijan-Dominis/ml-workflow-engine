import joblib
import json

from datetime import datetime

def save_pipeline_and_metadata(pipeline, cfg):
    # Step 1 - Save the trained model pipeline
    model_file = f"ml/models/trained/{cfg['name']}_{cfg['version']}.joblib"
    joblib.dump(pipeline, model_file)

    # Step 2 - Save metadata
    today = datetime.now().strftime("%Y-%m-%d")

    metadata = {
        "model_name": f"{cfg['name']}_{cfg['version']}",
        "task": cfg["task"],
        "trained_on": today,
        "features_version": cfg["data"]["features_version"],
        "model_class": cfg["model"]["class"]
    }

    metadata_file = model_file.replace("trained", "metadata").replace(".joblib", ".json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    # Step 3 - Print success message
    print(f"Pipeline and metadata saved successfully for model {cfg['name']}_{cfg['version']}.")