"""Abstract interfaces for feature freezing strategies."""

from abc import ABC, abstractmethod

from ml.config.compute_data_config_hash import compute_data_config_hash
from ml.feature_freezing.constants.output import FreezeOutput
from ml.feature_freezing.freeze_strategies.tabular.config.models import TabularFeaturesConfig


class FreezeStrategy(ABC):
    """Base strategy contract for feature snapshot generation."""

    @abstractmethod
    def freeze(
        self,
        config: TabularFeaturesConfig,
        *,
        timestamp: str,
        snapshot_id: str,
        snapshot_binding_key: str | None,
        start_time: float,
        owner: str
    ) -> FreezeOutput:
        """Execute strategy-specific freeze workflow and return output payload.

        Args:
            config: Feature-freezing configuration.
            timestamp: Snapshot creation timestamp.
            snapshot_id: Snapshot identifier.
            snapshot_binding_key: Optional key for a snapshot binding to define which snapshot to load for each dataset.
            start_time: Monotonic start timestamp.
            owner: Snapshot owner identifier.

        Returns:
            FreezeOutput: Freeze operation output payload.
        """
        pass

    @staticmethod
    def hash_config(config: TabularFeaturesConfig) -> str:
        """Compute deterministic hash for strategy configuration object.

        Args:
            config: Feature-freezing configuration object.

        Returns:
            str: Deterministic config hash.
        """
        return compute_data_config_hash(config)
