# General imports
import logging
from pathlib import Path

import joblib

from ml.exceptions import PersistenceError

logger = logging.getLogger(__name__)

def save_model(model, path: Path) -> str:
    model_file = path/f"model.joblib"
    try:
        joblib.dump(model, model_file)
        logger.info(f"Model successfully saved to {model_file}.")
        return str(model_file)
    except Exception as e:
        msg = f"Failed to save model to {model_file}."
        logger.exception(msg)
        raise PersistenceError(msg) from e