"""Module for validating runtime information against expected metadata to ensure reproducibility."""
import logging

from ml.exceptions import RuntimeMLError
from ml.modeling.models.runtime_info import RuntimeInfo

logger = logging.getLogger(__name__)

def validate_runtime_info(runtime_info_dict: dict) -> RuntimeInfo:
    """
    Validate runtime information dictionary against the RuntimeInfo model.

    Args:
        runtime_info_dict: Dictionary containing runtime information to validate.

    Returns:
        An instance of RuntimeInfo if validation is successful.
    """
    try:
        runtime_info = RuntimeInfo(**runtime_info_dict)
        logger.debug(f"Validated runtime info: {runtime_info}")
        return runtime_info
    except Exception as e:
        msg = "Runtime info validation failed."
        logger.exception(msg)
        raise RuntimeMLError(msg) from e
