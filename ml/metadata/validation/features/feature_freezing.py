import logging
from typing import Any

from ml.exceptions import RuntimeMLError
from ml.metadata.schemas.features.feature_freezing import FreezeMetadata

logger = logging.getLogger(__name__)

def validate_freeze_metadata(metadata_dict: dict[str, Any]) -> FreezeMetadata:
    """Validate freeze metadata dictionary against the FreezeMetadata Pydantic model.

    Args:
        metadata_dict: Dictionary containing freeze metadata to be validated.

    Returns:
        An instance of FreezeMetadata if validation is successful.

    Raises:
        RuntimeMLError: If the input dictionary does not conform to the FreezeMetadata schema.
    """

    try:
        freeze_metadata = FreezeMetadata.model_validate(metadata_dict)
        logger.debug("Freeze metadata validation successful.")
        return freeze_metadata
    except Exception as e:
        msg = "Freeze metadata validation failed."
        logger.exception(msg)
        raise RuntimeMLError(msg) from e
