from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseSearcher(ABC):
    @abstractmethod
    def search(self, model_cfg) -> Dict[str, Any]: ...