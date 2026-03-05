import logging

from ml.exceptions import RuntimeMLError
from ml.modeling.models.runtime_info import RuntimeInfo

logger = logging.getLogger(__name__)

def validate_runtime_info(runtime_info_raw: dict) -> RuntimeInfo:
    """Validate raw runtime info payload.

    Args:
        runtime_info_raw: Raw runtime info payload.

    Returns:
        RuntimeInfo: Validated runtime info.
    """
    try:
        runtime_info = RuntimeInfo(**runtime_info_raw)
        logger.debug("Successfully validated runtime info payload.")
    except Exception as e:
        msg = "Invalid runtime info payload."
        logger.exception(msg)
        raise RuntimeMLError(msg) from e

    return runtime_info
