import logging
from pathlib import Path

import pandas as pd

from ml.exceptions import DataError
from ml.registry.hash_registry import hash_file
from ml.utils.features.hashing.hash_dataframe_content import \
    hash_dataframe_content
from ml.utils.features.hashing.hash_feature_schema import hash_feature_schema

logger = logging.getLogger(__name__)

def validate_feature_set(
    feature_set: pd.DataFrame,
    *, 
    metadata: dict, 
    file_path: Path, 
    strict: bool = True
) -> None:
    if "row_id" not in feature_set.columns:
        msg = f"Feature set loaded from {file_path} is missing required 'row_id' column."
        logger.error(msg)
        raise DataError(msg)

    expected_schema_hash = metadata["feature_schema_hash"]

    actual_schema_hash = hash_feature_schema(feature_set)

    if actual_schema_hash != expected_schema_hash:
        msg = f"Feature schema hash mismatch: expected {expected_schema_hash}, got {actual_schema_hash}"
        logger.error(msg)
        raise DataError(msg)

    if strict:
        expected_in_memory_hash = metadata["in_memory_hash"]
        actual_in_memory_hash = hash_dataframe_content(feature_set)
        if actual_in_memory_hash != expected_in_memory_hash:
            msg = f"In-memory feature hash mismatch: expected {expected_in_memory_hash}, got {actual_in_memory_hash}"
            logger.warning(msg)

        expected_file_hash = metadata["file_hash"]
        actual_file_hash = hash_file(file_path)
        if actual_file_hash != expected_file_hash:
            msg = f"File hash mismatch: expected {expected_file_hash}, got {actual_file_hash}"
            logger.error(msg)
            raise DataError(msg)
            