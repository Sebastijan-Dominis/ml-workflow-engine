import hashlib
import logging
from pathlib import Path

import pandas as pd

from ml.exceptions import DataError, UserError
from ml.utils.loaders import read_data
from ml.registry.hash_registry import HASH_LOADER_REGISTRY

logger = logging.getLogger(__name__)

def load_data_with_loader_validation_hash(path: Path, format: str) -> tuple[pd.DataFrame, str]:
    if format not in HASH_LOADER_REGISTRY:
        msg = f"Unsupported data format for loading and hashing: {format}"
        logger.error(msg)
        raise UserError(msg)

    loader_validation_hash = HASH_LOADER_REGISTRY[format](path)
    
    data = read_data(format, path)
    return data, loader_validation_hash