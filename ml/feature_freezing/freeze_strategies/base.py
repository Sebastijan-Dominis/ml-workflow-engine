from abc import ABC, abstractmethod
from typing import Tuple

class FreezeStrategy(ABC):
    @abstractmethod
    def freeze(self, context, config) -> Tuple:
        pass