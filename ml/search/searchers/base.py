from abc import ABC, abstractmethod
from typing import Any, Dict

from ml.config.validation_schemas.model_cfg import SearchModelConfig


class BaseSearcher(ABC):
    @abstractmethod
    def search(self, model_cfg: SearchModelConfig) -> tuple[Dict[str, Any], list[dict], str]: ...