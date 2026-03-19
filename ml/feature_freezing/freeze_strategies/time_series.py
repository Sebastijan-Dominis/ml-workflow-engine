"""Placeholder strategy module for future time-series feature freezing."""

import logging

from ml.feature_freezing.constants.output import FreezeOutput
from ml.feature_freezing.freeze_strategies.base import FreezeStrategy

logger = logging.getLogger(__name__)

class FreezeTimeSeries(FreezeStrategy):
    """Stub strategy for time-series feature freezing workflows."""

    def freeze(
        self,
        config,
        *,
        snapshot_id: str,
        timestamp: str,
        start_time: float,
        owner: str,
        snapshot_binding_key: str | None
    ) -> FreezeOutput:
        """Execute time-series freeze workflow (not yet implemented).

        Args:
            config: Time-series feature-freezing configuration.
            timestamp: Run timestamp string used for metadata and paths.
            snapshot_id: Unique snapshot identifier.
            start_time: Process start time used for runtime metadata.
            owner: Owner identifier stored in snapshot metadata.
            snapshot_binding_key: Optional key for a snapshot binding to define which snapshot to load for each dataset.
        Returns:
            Freeze output containing persisted snapshot path and metadata.
        """
        raise NotImplementedError("Time-series freeze not implemented yet") # To be implemented in the future
