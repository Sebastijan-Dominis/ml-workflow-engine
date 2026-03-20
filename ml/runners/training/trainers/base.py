"""Protocol definition for model training runner implementations."""

from abc import abstractmethod
from pathlib import Path
from typing import Protocol

from ml.config.schemas.model_cfg import TrainModelConfig
from ml.runners.training.constants.output import TrainOutput


class Trainer(Protocol):
    """Structural interface implemented by concrete model trainers."""

    @abstractmethod
    def train(
        self,
        model_cfg: TrainModelConfig,
        *,
        snapshot_binding_key: str | None = None,
        strict: bool,
        failure_management_dir: Path,
        search_dir: Path
    ) -> TrainOutput:
        """Train model artifacts using provided configuration and runtime controls.

        Args:
            model_cfg: Validated training model configuration.
            strict: Whether data/validation loading should fail strictly.
            failure_management_dir: Directory for failure-management artifacts.
            snapshot_binding_key: Optional key for a snapshot binding to define which snapshot to load for each dataset.
        Returns:
            Standardized training output.
        """
        ...
