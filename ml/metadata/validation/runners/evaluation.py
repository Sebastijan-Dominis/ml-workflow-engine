import logging
from typing import Any

from ml.exceptions import UserError
from ml.metadata.schemas.runners.evaluation import EvaluationMetadata

logger = logging.getLogger(__name__)
def validate_evaluation_metadata(metadata_dict: dict[str, Any]) -> EvaluationMetadata:
    """Validate evaluation metadata dictionary against the EvaluationMetadata Pydantic model.

    Args:
        metadata_dict: Dictionary containing evaluation metadata to be validated.

    Returns:
        An instance of EvaluationMetadata if validation is successful.

    Raises:
        pydantic.ValidationError: If the input dictionary does not conform to the EvaluationMetadata schema.
    """

    try:
        evaluation_metadata = EvaluationMetadata(**metadata_dict)
        logger.debug("Evaluation metadata validation successful.")
        return evaluation_metadata
    except Exception as e:
        msg = "Evaluation metadata validation failed."
        logger.exception(msg)
        raise UserError(msg) from e
