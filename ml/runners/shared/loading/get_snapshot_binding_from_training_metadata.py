"""Helpers for extracting feature snapshot bindings from training metadata."""

import logging

from ml.exceptions import DataError
from ml.metadata.schemas.runners.training import TrainingMetadata
from ml.modeling.models.feature_lineage import FeatureLineage

logger = logging.getLogger(__name__)

def get_snapshot_binding_from_training_metadata(training_metadata: TrainingMetadata) -> list[FeatureLineage]:
    """Return feature lineage snapshot bindings required for consistent loading.

    Args:
        training_metadata: Training metadata object containing lineage information.

    Returns:
        list of FeatureLineage objects representing the snapshot binding.
    """

    snapshot_binding = training_metadata.lineage.feature_lineage
    if snapshot_binding is None:
        msg = "No snapshot binding found in training metadata lineage. Cannot proceed without knowing which feature snapshots were used during training."
        logger.error(msg)
        raise DataError(msg)
    return snapshot_binding
