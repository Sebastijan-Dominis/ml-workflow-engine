import logging

from ml.exceptions import RuntimeMLError
from ml.metadata.schemas.runners.evaluation import EvaluationArtifacts
from ml.metadata.schemas.runners.explainability import ExplainabilityArtifacts

logger = logging.getLogger(__name__)

def validate_evaluation_artifacts(evaluation_artifacts_raw: dict) -> EvaluationArtifacts:
    """Validate the evaluation artifacts by attempting to construct the EvaluationArtifacts model.

    Args:
        evaluation_artifacts_raw: Raw dictionary of evaluation artifact paths and hashes.

    Returns:
        EvaluationArtifacts: Validated evaluation artifacts model.

    Raises:
        RuntimeMLError: If validation fails due to missing required fields or incorrect types.
    """
    try:
        evaluation_artifacts = EvaluationArtifacts(**evaluation_artifacts_raw)
        return evaluation_artifacts
    except Exception as e:
        msg = "Failed to construct evaluation artifacts model."
        logger.exception(msg)
        raise RuntimeMLError(msg) from e

def validate_explainability_artifacts(explainability_artifacts_raw: dict) -> ExplainabilityArtifacts:
    """Validate the explainability artifacts by attempting to construct the ExplainabilityArtifacts model.

    Args:
        explainability_artifacts_raw: Raw dictionary of explainability artifact paths and hashes.

    Returns:
        ExplainabilityArtifacts: Validated explainability artifacts model.

    Raises:
        RuntimeMLError: If validation fails due to missing required fields or incorrect types.
    """
    try:
        explainability_artifacts = ExplainabilityArtifacts(**explainability_artifacts_raw)
        return explainability_artifacts
    except Exception as e:
        msg = "Failed to construct explainability artifacts model."
        logger.exception(msg)
        raise RuntimeMLError(msg) from e
