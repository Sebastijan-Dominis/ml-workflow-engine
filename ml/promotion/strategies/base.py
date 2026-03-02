"""Abstract strategy interface for stage-specific promotion behavior."""

from abc import ABC, abstractmethod

from ml.promotion.context import PromotionContext
from ml.promotion.result import PromotionResult
from ml.promotion.state import PromotionState


class PromotionStrategy(ABC):
    """Base contract for promotion strategy implementations."""

    @abstractmethod
    def execute(
        self,
        context: PromotionContext,
        state: PromotionState,
    ) -> PromotionResult:
        """Execute stage-specific promotion logic and return result payload.

        Args:
            context: Promotion runtime context.
            state: Loaded promotion state.

        Returns:
            Promotion decision result.
        """
        ...