"""Protocol definition for hyperparameter searcher implementations."""

from abc import abstractmethod
from pathlib import Path
from typing import Protocol

from ml.config.schemas.model_cfg import SearchModelConfig
from ml.search.searchers.output import SearchOutput


class Searcher(Protocol):
    """Structural interface implemented by concrete search backends."""

    @abstractmethod
    def search(
        self,
        model_cfg: SearchModelConfig,
        *,
        snapshot_binding_key: str | None = None,
        strict: bool,
        failure_management_dir: Path
    ) -> SearchOutput:
        """Run hyperparameter search and return standardized search artifacts.

        Args:
            model_cfg: Validated search model configuration.
            strict: Whether loading and validation should fail strictly.
            failure_management_dir: Directory for failure-management artifacts.

        Returns:
            Standardized search output.
        """
        ...
