"""Runtime platform information collection utilities."""

import logging
import platform

import psutil
from ml.exceptions import RuntimeMLError

logger = logging.getLogger(__name__)

def get_runtime_info() -> dict:
    """Collect OS, hardware, and Python runtime characteristics.

    Args:
        None.

    Returns:
        Dictionary containing runtime platform characteristics.
    """

    try:
        ram_total_gb = round(psutil.virtual_memory().total / 1e9, 2)
        runtime_info = {
            "os": platform.system(),
            "os_release": platform.release(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "ram_total_gb": ram_total_gb,
            "platform_string": platform.platform(),
            "hostname": platform.node(),
            "python_version": platform.python_version(),
            "python_impl": platform.python_implementation(),
            "python_build": platform.python_build(),
        }
        logger.debug(f"Collected runtime info: {runtime_info}")
        return runtime_info
    except Exception as e:
        msg = f"Failed to get runtime info: {e}"
        logger.error(msg)
        raise RuntimeMLError(msg) from e
