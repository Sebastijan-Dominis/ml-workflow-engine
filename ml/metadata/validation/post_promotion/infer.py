"""A module for validating inference metadata in the inference pipeline."""
import logging

from ml.exceptions import RuntimeMLError
from ml.metadata.schemas.post_promotion.infer import InferenceMetadata

logger = logging.getLogger(__name__)

def validate_inference_metadata(metadata: dict) -> InferenceMetadata:
    """
    Validate the inference metadata against the InferenceMetadata schema.

    Args:
        metadata (dict): The metadata dictionary to validate.

    Returns:
        InferenceMetadata: The validated metadata object.
    """
    try:
        validated_metadata = InferenceMetadata.model_validate(metadata)
        logger.debug("Successfully validated inference metadata.")
        return validated_metadata
    except Exception as e:
        msg = "Error validating inference metadata."
        logger.exception(msg)
        raise RuntimeMLError(msg) from e
