# General imports
import logging
logger = logging.getLogger(__name__)
from pathlib import Path
from datetime import datetime
import json

def save_metadata(cfg: dict):
    model_name = f"{cfg['problem']}_{cfg['segment']['name']}_{cfg['version']}"

    path = Path(f"ml/artifacts/{cfg['problem']}/{cfg['segment']['name']}/{cfg['version']}")

    # Step 1 - Ensure the path for saving the model exist
    path.mkdir(parents=True, exist_ok=True)

    # Step 2 - Save metadata
    today = datetime.now().strftime("%Y-%m-%d")

    metadata = {
        "model_name": f"{model_name}",
        "task": cfg["task"],
        "pipeline": cfg["pipeline"],
        "trained_on": today,
    }

    # Step 2.1 - Warn if target metadata file already exists
    metadata_file = Path(path)/f"metadata.json"
    if metadata_file.exists():
        logger.warning(
            f"Metadata file for model {model_name} "
            "already exists and will be overwritten."
        )

    # Step 2.2 - Write metadata to JSON file
    try:
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Metadata for model {model_name} successfully saved to {metadata_file}.")
    except Exception:
        logger.exception(f"Error saving metadata to {metadata_file}")
        raise