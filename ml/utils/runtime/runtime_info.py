import logging
import platform

import psutil

from ml.exceptions import RuntimeMLException

logger = logging.getLogger(__name__)

def get_runtime_info() -> dict:
    try:
        ram_total_gb = round(psutil.virtual_memory().total / 1e9, 2)
        return {
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
    except Exception as e:
        msg = f"Failed to get runtime info: {e}"
        logger.error(msg)
        raise RuntimeMLException(msg)