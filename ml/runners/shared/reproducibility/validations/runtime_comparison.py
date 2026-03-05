"""Reproducibility checks comparing runtime platform characteristics."""

import logging
import platform

from ml.modeling.models.runtime_info import RuntimeInfo

logger = logging.getLogger(__name__)

def validate_runtime_matches(runtime_info: RuntimeInfo) -> None:
    """Compare current runtime details with expected metadata and log mismatches.

    Args:
        runtime_info: Runtime metadata dictionary with expected platform details.

    Returns:
        None.
    """

    python_version = platform.python_version()
    expected_python_version = runtime_info.runtime.python_version
    if expected_python_version != python_version:
        logger.warning(
            f"Current Python version {python_version} does not match expected {expected_python_version}. "
            "Reproducibility may be affected."
        )
    else:
        logger.debug("Python version matches expected.")

    os = platform.system()
    expected_os = runtime_info.runtime.os
    if expected_os != os:
        logger.warning(
            f"Current operating system {os} does not match expected {expected_os}. "
            "Reproducibility may be affected."
        )
    else:
        logger.debug("Operating system matches expected.")

    cpu = platform.processor()
    expected_cpu = runtime_info.runtime.processor
    if expected_cpu != cpu:
        logger.warning(
            f"Current CPU {cpu} does not match expected {expected_cpu}. "
            "Reproducibility may be affected."
        )
    else:
        logger.debug("CPU matches expected.")

    os_release = platform.release()
    expected_os_release = runtime_info.runtime.os_release
    if expected_os_release != os_release:
        logger.warning(
            f"Current OS release {os_release} does not match expected {expected_os_release}. "
            "Reproducibility may be affected."
        )
    else:
        logger.debug("OS release matches expected.")
