# General imports
import logging
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline

from ml.exceptions import PersistenceError

logger = logging.getLogger(__name__)

def save_pipeline(pipeline: Pipeline, path: Path) -> str:
    pipeline_file = path / "pipeline.joblib"
    
    try:
        joblib.dump(pipeline, pipeline_file)
        logger.info(f"Pipeline successfully saved to {pipeline_file}.")
        return str(pipeline_file)
    except Exception as e:
        msg = f"Failed to save pipeline to {pipeline_file}."
        logger.exception(msg)
        raise PersistenceError(msg) from e