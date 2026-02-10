from abc import abstractmethod
from typing import Any, Protocol

from ml.config.validation_schemas.model_cfg import SearchModelConfig


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
    """
    @abstractmethod
    def search(self, model_cfg: SearchModelConfig, strict: bool) -> tuple[dict[str, Any], list[dict], str]: ...