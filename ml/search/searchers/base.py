from abc import abstractmethod
from typing import Any, Protocol

from ml.config.validation_schemas.model_cfg import SearchModelConfig
from ml.search.searchers.output import SearchOutput


class Searcher(Protocol):
    """
    Searcher interface.

    Returns:
        tuple containing:
        - dict with keys:
            - best_params
            - phases
        - list of dicts representing feature lineage
        - str representing pipeline hash
        - str representing scoring method
    """
    @abstractmethod
    def search(self, model_cfg: SearchModelConfig, strict: bool) -> SearchOutput: ...