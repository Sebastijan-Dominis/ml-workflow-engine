"""Helpers for extracting feature snapshot bindings from training metadata."""

import logging

from ml.exceptions import DataError

logger = logging.getLogger(__name__)

def get_snapshot_binding_from_training_metadata(train_metadata: dict) -> list[dict]:
    """Return feature lineage snapshot bindings required for consistent loading.

    Args:
        train_metadata: Training metadata dictionary containing lineage information.

    Returns:
        List of feature snapshot binding entries.
    """

    snapshot_binding = train_metadata.get("lineage", {}).get("feature_lineage")
    if snapshot_binding is None:
        msg = "No snapshot binding found in training metadata lineage. Cannot proceed without knowing which feature snapshots were used during training."
        logger.error(msg)
        raise DataError(msg)
    return snapshot_binding