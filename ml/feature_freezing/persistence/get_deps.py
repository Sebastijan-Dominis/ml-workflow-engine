"""Dependency version collection utilities for freeze runtime metadata."""

import logging
from importlib.metadata import version

from ml.exceptions import RuntimeMLError

logger = logging.getLogger(__name__)

def get_pkg_version(name):
    """Return installed package version for the provided package name.

    Args:
        name: Package name.

    Returns:
        str: Installed package version.
    """

    return version(name)

def get_deps():
    """Collect key runtime dependency versions for metadata persistence.

    Returns:
        dict: Dependency versions keyed by package name.
    """

    try:
        deps = {
            "numpy": get_pkg_version("numpy"),
            "pandas": get_pkg_version("pandas"),
            "scikit_learn": get_pkg_version("scikit-learn"),
            "pyarrow": get_pkg_version("pyarrow"),
            "pydantic": get_pkg_version("pydantic"),
            "PyYAML": get_pkg_version("PyYAML"),
        }
    except Exception as e:
        msg = f"Failed to get package versions: {e}"
        logger.error(msg)
        raise RuntimeMLError(msg) from e

    logger.debug(f"Collected dependencies: {deps}")
    return deps
