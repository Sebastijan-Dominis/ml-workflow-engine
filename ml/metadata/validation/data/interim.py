"""Module for validating interim dataset metadata."""
import logging

from ml.exceptions import RuntimeMLError
from ml.metadata.schemas.data.interim import InterimDatasetMetadata

logger = logging.getLogger(__name__)

def validate_interim_dataset_metadata(metadata: dict) -> InterimDatasetMetadata:
    """
    Validate the interim dataset metadata against the InterimDatasetMetadata schema.

    Args:
        metadata (dict): The metadata dictionary to validate.

    Returns:
        InterimDatasetMetadata: The validated metadata object.
    """
    try:
        validated_metadata = InterimDatasetMetadata.model_validate(metadata)
        logger.debug("Successfully validated interim dataset metadata.")
        return validated_metadata
    except Exception as e:
        msg = "Error validating interim dataset metadata."
        logger.exception(msg)
        raise RuntimeMLError(msg) from e
