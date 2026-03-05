"""Module for validating raw snapshot metadata against the RawSnapshotMetadata schema."""
import logging

from ml.exceptions import RuntimeMLError
from ml.metadata.schemas.data.raw import RawSnapshotMetadata

logger = logging.getLogger(__name__)

def validate_raw_snapshot_metadata(metadata: dict) -> RawSnapshotMetadata:
    """
    Validate the raw snapshot metadata against the RawSnapshotMetadata schema.

    Args:
        metadata (dict): The metadata dictionary to validate.
    Returns:
        RawSnapshotMetadata: The validated metadata object.
    """
    try:
        validated_metadata = RawSnapshotMetadata.model_validate(metadata)
        logger.debug("Successfully validated raw snapshot metadata.")
        return validated_metadata
    except Exception as e:
        msg = "Error validating raw snapshot metadata."
        logger.exception(msg)
        raise RuntimeMLError(msg) from e
