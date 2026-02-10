import logging
from pathlib import Path

import joblib

from ml.runners.evaluation.evaluators.classification.classes import ProbabilisticClassifier
from ml.exceptions import PipelineContractError

logger = logging.getLogger(__name__)

def load_pipeline(pipeline_file: Path) -> ProbabilisticClassifier:
    """Load a serialized pipeline from disk.

    Args:
        pipeline_file (pathlib.Path): Path to the serialized pipeline file.

    Returns:
        ProbabilisticClassifier: Deserialized model/estimator.
    """
    if not pipeline_file.exists():
        msg = f"Pipeline file not found at {pipeline_file}"
        logger.error(msg)
        raise PipelineContractError(msg)
    
    with open(pipeline_file, "rb") as f:
        pipeline = joblib.load(f)
    
    return pipeline