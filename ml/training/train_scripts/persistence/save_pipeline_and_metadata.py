"""Persistence helpers for saving trained models and associated metadata.

This module provides a single helper function, ``save_pipeline_and_metadata``,
which persists a trained sklearn ``Pipeline`` using joblib and writes a small
JSON metadata file describing the training run. The function ensures target
directories exist and logs any errors encountered while saving.
"""

# General imports
import logging
logger = logging.getLogger(__name__)
import joblib
import json

from sklearn.pipeline import Pipeline
from pathlib import Path
from datetime import datetime

# File-system constants
from ml.training.train_scripts.persistence.constants import (
    model_dir,
    metadata_dir,
)


def save_pipeline_and_metadata(pipeline: Pipeline, cfg: dict):
    """Save a trained pipeline and its metadata to disk.

    The function writes a joblib file for the provided ``pipeline`` and a
    small JSON file containing basic metadata (model identifier, task,
    training date and features version). Any IO or serialization exceptions
    are logged and re-raised for the caller to handle.

    Args:
        pipeline (Pipeline): A fitted sklearn pipeline to persist.
        cfg (dict): Validated configuration dictionary used for training.
    """

    # Step 1 - Ensure the paths for saving the model and metadata exist
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    Path(metadata_dir).mkdir(parents=True, exist_ok=True)

    # Step 2 - Save the trained model pipeline
    model_file = Path(model_dir)/f"{cfg['name']}_{cfg['version']}.joblib"

    try:
        joblib.dump(pipeline, model_file)
    except Exception:
        logger.exception(f"Error saving model pipeline to {model_file}")
        raise

    # Step 3 - Save metadata
    today = datetime.now().strftime("%Y-%m-%d")

    metadata = {
        "model_name": f"{cfg['name']}_{cfg['version']}",
        "task": cfg["task"],
        "trained_on": today,
        "features_version": cfg["data"]["features_version"],
    }

    metadata_file = Path(metadata_dir)/f"{cfg['name']}_{cfg['version']}.json"
    try:
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
    except Exception:
        logger.exception(f"Error saving metadata to {metadata_file}")
        raise

    # Step 4 - Log success message
    logger.info(
        f"Pipeline and metadata saved successfully for model {cfg['name']}_{cfg['version']}."
    )