from abc import ABC, abstractmethod
from ml.promotion.state import PromotionState
from ml.promotion.context import PromotionContext
from ml.promotion.result import PromotionResult


class PromotionStrategy(ABC):

    @abstractmethod
    def execute(
        self,
        context: PromotionContext,
        state: PromotionState,
    ) -> PromotionResult:
        ...