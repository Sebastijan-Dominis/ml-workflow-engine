import logging

from ml.exceptions import RuntimeMLError
from ml.promotion.config.registry_entry import RegistryEntry

logger = logging.getLogger(__name__)

def validate_registry_entry(entry_raw: dict) -> RegistryEntry:
    """
    Validate the raw registry entry against the RegistryEntry schema.

    Args:
        entry_raw (dict): The raw registry entry dictionary to validate.

    Returns:
        RegistryEntry: The validated registry entry object.
    """
    try:
        validated_entry = RegistryEntry.model_validate(entry_raw)
        logger.debug("Successfully validated raw registry entry.")
        return validated_entry
    except Exception as e:
        msg = "Error validating raw registry entry."
        logger.exception(msg)
        raise RuntimeMLError(msg) from e
