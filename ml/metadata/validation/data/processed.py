"""Module for validating processed dataset metadata."""
import logging

from ml.exceptions import RuntimeMLError
from ml.metadata.schemas.data.processed import ProcessedDatasetMetadata

logger = logging.getLogger(__name__)

def validate_processed_dataset_metadata(metadata: dict) -> ProcessedDatasetMetadata:
    """
    Validate the processed dataset metadata against the ProcessedDatasetMetadata schema.

    Args:
        metadata (dict): The metadata dictionary to validate.
    Returns:
        ProcessedDatasetMetadata: The validated metadata object.
    """
    try:
        validated_metadata = ProcessedDatasetMetadata.model_validate(metadata)
        logger.debug("Successfully validated processed dataset metadata.")
        return validated_metadata
    except Exception as e:
        msg = "Error validating processed dataset metadata."
        logger.exception(msg)
        raise RuntimeMLError(msg) from e
