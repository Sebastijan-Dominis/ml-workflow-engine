"""Small utilities used by evaluation scripts.

Contains simple helpers for validating configuration dictionaries.
"""

import logging
logger = logging.getLogger(__name__)


def assert_keys(d: dict, keys: list[str]) -> None:
    """Assert that all `keys` are present in mapping `d`.

    Args:
        d (dict): Mapping to check.
        keys (list[str]): Keys that must be present in `d`.

    Raises:
        KeyError: If any required key is missing. The missing key is logged
            and returned in the exception message to aid debugging.
    """

    for k in keys:
        if k not in d:
            msg = f"Missing key '{k}'. Current keys: {list(d.keys())}"
            logger.error(msg)
            raise KeyError(msg)