"""Model artifact persistence helpers for training runs."""

import logging
import os
import tempfile
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
    model_file.parent.mkdir(parents=True, exist_ok=True)

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=model_file.parent,
            prefix="model.",
            suffix=".joblib.tmp",
            delete=False,
        ) as tmp_file:
            temp_path = Path(tmp_file.name)

        joblib.dump(model, temp_path)
        os.replace(temp_path, model_file)

        msg = f"Model successfully saved to {model_file}."
        logger.info(msg)
        print(msg)
        return model_file
    except Exception as e:
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                logger.warning("Failed to clean up temporary model file: %s", temp_path)

        msg = f"Failed to save model to {model_file}."
        logger.exception(msg)
        raise PersistenceError(msg) from e
