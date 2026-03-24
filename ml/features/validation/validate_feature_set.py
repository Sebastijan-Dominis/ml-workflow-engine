"""Validation utilities for feature-set schema and integrity hash checks."""

import logging
from pathlib import Path

import pandas as pd

from ml.exceptions import DataError
from ml.features.hashing.hash_dataframe_content import hash_dataframe_content
from ml.features.hashing.hash_feature_schema import hash_feature_schema
from ml.features.loading.schemas import load_feature_set_schemas
from ml.utils.hashing.service import hash_file

logger = logging.getLogger(__name__)

def validate_feature_set(
    feature_set: pd.DataFrame,
    *,
    metadata: dict,
    file_path: Path,
    strict: bool = True
) -> None:
    """Validate required columns and metadata hash consistency for a feature set.

    Args:
        feature_set: Feature dataframe to validate.
        metadata: Snapshot metadata containing expected hashes.
        file_path: Feature file path used for file-hash validation.
        strict: Whether to enforce strict in-memory and file hash checks.

    Returns:
        None.
    """

    entity_key = metadata.get("entity_key")
    if entity_key not in feature_set.columns:
        msg = f"Feature set loaded from {file_path} is missing required 'entity_key' column."
        logger.error(msg)
        raise DataError(msg)

    expected_schema_hash = metadata["feature_schema_hash"]
    actual_schema_hash = hash_feature_schema(feature_set)

    if len(file_path.parents) < 2:
        msg = f"File path {file_path} is too short to derive schema/operator info for validation."
        logger.error(msg)
        raise DataError(msg)
    feature_set_version_path = file_path.parents[1]
    # Internally validates operator hashes; underscore variables to indicate the purpose is solely hash validation, not schema loading
    _, _ = load_feature_set_schemas(feature_set_version_path, file_path.parent)

    if actual_schema_hash != expected_schema_hash:
        msg = f"Feature schema hash mismatch for {file_path}: expected {expected_schema_hash}, got {actual_schema_hash}"
        logger.error(msg)
        raise DataError(msg)

    if strict:
        expected_in_memory_hash = metadata["in_memory_hash"]
        actual_in_memory_hash = hash_dataframe_content(feature_set)
        if actual_in_memory_hash != expected_in_memory_hash:
            msg = f"In-memory feature hash mismatch for {file_path}: expected {expected_in_memory_hash}, got {actual_in_memory_hash}"
            logger.warning(msg)

        expected_file_hash = metadata["file_hash"]
        actual_file_hash = hash_file(file_path)
        if actual_file_hash != expected_file_hash:
            msg = f"File hash mismatch for {file_path}: expected {expected_file_hash}, got {actual_file_hash}"
            logger.error(msg)
            raise DataError(msg)
