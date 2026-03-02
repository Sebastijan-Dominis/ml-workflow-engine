from abc import abstractmethod
from pathlib import Path
from typing import Protocol

from ml.config.validation_schemas.model_cfg import SearchModelConfig
from ml.search.searchers.output import SearchOutput


class Searcher(Protocol):
    @abstractmethod
    def search(
        self, 
        model_cfg: SearchModelConfig, 
        *,
        strict: bool,
        failure_management_dir: Path
    ) -> SearchOutput: ...