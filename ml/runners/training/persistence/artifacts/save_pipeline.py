"""Pipeline artifact persistence helpers for training runs."""

import logging
import os
import tempfile
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline

from ml.exceptions import PersistenceError

logger = logging.getLogger(__name__)

def save_pipeline(pipeline: Pipeline, path: Path) -> Path:
    """Persist trained preprocessing+model pipeline and return file path.

    Args:
        pipeline: Trained sklearn pipeline to serialize.
        path: Target directory where the pipeline artifact is saved.

    Returns:
        Filesystem path to the persisted pipeline artifact.
    """

    pipeline_file = path / "pipeline.joblib"
    pipeline_file.parent.mkdir(parents=True, exist_ok=True)

    temp_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=pipeline_file.parent,
            prefix="pipeline.",
            suffix=".joblib.tmp",
            delete=False,
        ) as tmp_file:
            temp_path = Path(tmp_file.name)

        joblib.dump(pipeline, temp_path)
        os.replace(temp_path, pipeline_file)

        msg = f"Pipeline successfully saved to {pipeline_file}."
        logger.info(msg)
        print(msg)
        return pipeline_file
    except Exception as e:
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                logger.warning("Failed to clean up temporary pipeline file: %s", temp_path)

        msg = f"Failed to save pipeline to {pipeline_file}."
        logger.exception(msg)
        raise PersistenceError(msg) from e
