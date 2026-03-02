"""Pipeline artifact persistence helpers for training runs."""

import logging
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
    
    try:
        joblib.dump(pipeline, pipeline_file)
        logger.info(f"Pipeline successfully saved to {pipeline_file}.")
        return pipeline_file
    except Exception as e:
        msg = f"Failed to save pipeline to {pipeline_file}."
        logger.exception(msg)
        raise PersistenceError(msg) from e