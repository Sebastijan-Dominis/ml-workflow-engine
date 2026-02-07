import logging
logger = logging.getLogger(__name__)
import platform
from ml.exceptions import RuntimeMLException

def get_runtime_info() -> dict:
    try:
        return {
            "python_version": platform.python_version(),
            "python_impl": platform.python_implementation(),
            "os": platform.system(),
            "os_release": platform.release(),
            "architecture": platform.machine(),
            "platform_string": platform.platform(),
            "id": platform.node(),
            "processor": platform.processor(),
            "python_build": platform.python_build(),
        }
    except Exception as e:
        msg = f"Failed to get runtime info: {e}"
        logger.error(msg)
        raise RuntimeMLException(msg)