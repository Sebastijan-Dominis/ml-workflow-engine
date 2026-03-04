import logging
from typing import Any

from ml.exceptions import UserError
from ml.metadata.schemas.runners.training import TrainingMetadata

logger = logging.getLogger(__name__)

def validate_training_metadata(training_metadata: dict[str, Any]) -> TrainingMetadata:
    """Validate the structure and content of training metadata.

    Args:
        training_metadata: The training metadata dictionary to validate.
    Returns:
        The validated TrainingMetadata object.
    Raises:
        UserError: If any required fields are missing or contain invalid values.
    """

    try:
        validated_metadata = TrainingMetadata(**training_metadata)
        logger.debug("Training metadata validation successful.")
        return validated_metadata
    except Exception as e:
        msg = "Training metadata validation failed."
        logger.exception(msg)
        raise UserError(msg) from e
