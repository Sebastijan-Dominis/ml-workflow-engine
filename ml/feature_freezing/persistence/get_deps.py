import logging
from importlib.metadata import version

from ml.exceptions import RuntimeMLException

logger = logging.getLogger(__name__)

def get_pkg_version(name):
    return version(name)

def get_deps():
    try:
        deps = {
            "numpy": get_pkg_version("numpy"),
            "pandas": get_pkg_version("pandas"),
            "scikit-learn": get_pkg_version("scikit-learn"),
            "pyarrow": get_pkg_version("pyarrow"),
            "pydantic": get_pkg_version("pydantic"),
            "PyYAML": get_pkg_version("PyYAML"),
        }
    except Exception as e:
        msg = f"Failed to get package versions: {e}"
        logger.error(msg)
        raise RuntimeMLException(msg)

    return deps