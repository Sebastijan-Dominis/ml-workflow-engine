import logging
from typing import Any

from ml.exceptions import UserError
from ml.metadata.schemas.runners.explainability import ExplainabilityMetadata

logger = logging.getLogger(__name__)

def validate_explainability_metadata(explainability_metadata: dict[str, Any]) -> ExplainabilityMetadata:
    """Validate the structure and content of explainability metadata.

    Args:
        explainability_metadata: The explainability metadata dictionary to validate.
    Returns:
        The validated ExplainabilityMetadata object.
    Raises:
        UserError: If any required fields are missing or contain invalid values.
    """

    try:
        validated_metadata = ExplainabilityMetadata(**explainability_metadata)
        logger.debug("Explainability metadata validation successful.")
        return validated_metadata
    except Exception as e:
        msg = "Explainability metadata validation failed."
        logger.exception(msg)
        raise UserError(msg) from e
