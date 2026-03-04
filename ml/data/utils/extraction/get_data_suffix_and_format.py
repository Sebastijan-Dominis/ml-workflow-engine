"""Helpers for resolving dataset path suffix and format from metadata."""

import logging
from typing import Literal

from ml.exceptions import UserError

logger = logging.getLogger(__name__)

def get_data_suffix_and_format(metadata: dict, location: Literal["data/output", "data"]) -> tuple[str, str]:
    """Extract required data path suffix and format fields for a metadata location.

    Args:
        metadata: Metadata dictionary containing data path and format fields.
        location: Metadata location key to resolve (``data/output`` or ``data``).

    Returns:
        Tuple of data path suffix and data format.
    """

    if location == "data/output":
        data_suffix = metadata.get("data", {}).get("output", {}).get("path_suffix")
        data_format = metadata.get("data", {}).get("output", {}).get("format")
    elif location == "data":
        data_suffix = metadata.get("data", {}).get("path_suffix")
        data_format = metadata.get("data", {}).get("format")
    else:
        msg = f"Invalid location '{location}' specified. Expected 'data/output' or 'data'."
        logger.error(msg)
        raise UserError(msg)

    if data_suffix is None:
        msg = f"Metadata is missing 'data.path_suffix' field, which is required to locate the data file. Metadata content: {metadata}"
        logger.error(msg)
        raise UserError(msg)
    
    if data_format is None:
        msg = f"Metadata is missing 'data.format' field, which is required to read the data file. Metadata content: {metadata}"
        logger.error(msg)
        raise UserError(msg)
    
    return data_suffix, data_format