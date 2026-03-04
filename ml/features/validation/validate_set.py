"""Validation helpers for set-level consistency across feature snapshots."""

import logging

from ml.exceptions import DataError

logger = logging.getLogger(__name__)

def validate_set(hash_type: str, hashes: set, feature_sets: list) -> None:
    """Ensure all feature sets share the same expected hash/category value.

    Args:
        hash_type: Hash/category label being validated.
        hashes: Set of observed hash values across feature sets.
        feature_sets: Feature-set descriptors used for error context.

    Returns:
        None.
    """

    if len(hashes) != 1:
        msg = f"{hash_type} hashes do not match across feature sets. Feature sets involved: " + ", ".join(
            [f"{feature_sets[i].name} (version: {feature_sets[i].version})" for i in range(len(feature_sets))]
        )
        logger.error(msg)
        raise DataError(msg)
