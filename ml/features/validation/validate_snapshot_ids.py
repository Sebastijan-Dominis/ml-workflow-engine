"""Validation helpers for expected versus actual feature snapshot IDs."""

import logging

from ml.exceptions import DataError

logger = logging.getLogger(__name__)

def validate_snapshot_ids(feature_lineage, snapshot_selection):
    """Validate that loaded feature lineage snapshot IDs match requested selection.

    Args:
        feature_lineage: Loaded feature lineage records.
        snapshot_selection: Snapshot selections expected from training metadata.

    Returns:
        None.
    """

    expected_snapshot_ids = [sel.get("snapshot_id") for sel in snapshot_selection]
    actual_snapshot_ids = [lineage.snapshot_id for lineage in feature_lineage]
    if not expected_snapshot_ids or not actual_snapshot_ids:
        msg = f"Missing snapshot IDs in either expected snapshot selection or actual feature lineage. Cannot validate snapshot consistency for evaluation.\nexpected_snapshot_ids: {expected_snapshot_ids}\nactual_snapshot_ids: {actual_snapshot_ids}"
        logger.error(msg)
        raise DataError(msg)
    if expected_snapshot_ids != actual_snapshot_ids:
        msg = f"The snapshot IDs of the loaded feature lineage do not match the expected snapshot IDs from training metadata. This may indicate a mismatch in feature snapshots used for evaluation vs training.\nexpected_snapshot_ids: {expected_snapshot_ids}\nactual_snapshot_ids: {actual_snapshot_ids}"
        logger.error(msg)
        raise DataError(msg)
