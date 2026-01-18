"""Utilities to update the project's global models registry.

The training pipeline writes per-model metadata and artifacts. This
module exposes ``update_general_config`` which merges the newly-trained
model information into the central ``configs/models.yaml`` registry used
by downstream consumers.
"""

# General imports
import logging
logger = logging.getLogger(__name__)
import yaml

from pathlib import Path

# File-system constants
from ml.training.train_scripts.persistence.constants import (
    model_dir,
    metadata_dir,
    explainability_dir,
)


def update_general_config(cfg: dict) -> None:
    """Merge a trained model entry into ``configs/models.yaml``.

    The function constructs a standardized entry for the newly-trained
    model (paths to artifacts, explainability files, and feature metadata),
    ensures required directories exist, and writes or updates the YAML
    registry. Existing entries for the same model key are updated with a
    warning logged.

    Args:
        cfg (dict): Validated configuration dictionary describing the model.
    """

    model_key = f"{cfg['name']}_{cfg['version']}"

    # Step 1 - Prepare general config structure
    general_config = {
        model_key: {
            "name": cfg["name"],
            "version": cfg["version"],
            "task": cfg["task"],
            "target": cfg["data"]["target"],
            "algorithm": cfg["model"]["algorithm"],
            "features": {
                "version": cfg["data"]["features_version"],
                "path": cfg["data"]["features_path"],
                "schema": cfg["data"]["features_path"] + "/schema.csv",
            },
            "artifacts": {
                "model": f"{model_dir}/{cfg['name']}_{cfg['version']}.joblib",
                "metadata": f"{metadata_dir}/{cfg['name']}_{cfg['version']}.json",
                "feature_importances": f"{explainability_dir}/{cfg['name']}_{cfg['version']}/feature_importances.csv",
                "shap_importances": f"{explainability_dir}/{cfg['name']}_{cfg['version']}/shap_importances.csv",
            },
            "explainability": {
                "feature_importance_method": cfg["explainability"]["feature_importance_method"],
                "shap_method": cfg["explainability"]["shap_method"],
            },
            "threshold": cfg["model"].get("threshold", 0.5),
        }
    }

    # Step 2 - Ensure directories exist
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    Path(f"{explainability_dir}/{cfg['name']}_{cfg['version']}").mkdir(parents=True, exist_ok=True)
    Path(metadata_dir).mkdir(parents=True, exist_ok=True)

    # Step 3 - Load existing general config
    general_config_path = Path("configs/models.yaml")
    general_config_path.parent.mkdir(parents=True, exist_ok=True)

    if general_config_path.exists():
        with open(general_config_path) as f:
            existing = yaml.safe_load(f) or {}
    else:
        existing = {}

    # Step 4 - Warn if overwriting existing config
    if model_key in existing:
        logger.warning(f"Overwriting existing config for {model_key}")

    # Step 5 - Update existing config with new model info
    existing.setdefault(model_key, {}).update(general_config[model_key])

    with open(general_config_path, "w") as f:
        yaml.safe_dump(existing, f, sort_keys=False)

    # Step 6 - Log success message
    logger.info(f"General config successfully updated with model {model_key}.")