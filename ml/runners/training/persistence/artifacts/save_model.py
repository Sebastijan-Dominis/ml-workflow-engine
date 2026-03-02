"""Model artifact persistence helpers for training runs."""

import logging
from pathlib import Path

import joblib

from ml.exceptions import PersistenceError

logger = logging.getLogger(__name__)

def save_model(model, path: Path) -> Path:
    """Persist trained model artifact using joblib and return file path.

    Args:
        model: Trained model object to serialize.
        path: Target directory where the model artifact is saved.

    Returns:
        Filesystem path to the persisted model artifact.
    """

    model_file = path / "model.joblib"
    try:
        joblib.dump(model, model_file)
        logger.info(f"Model successfully saved to {model_file}.")
        return model_file
    except Exception as e:
        msg = f"Failed to save model to {model_file}."
        logger.exception(msg)
        raise PersistenceError(msg) from e