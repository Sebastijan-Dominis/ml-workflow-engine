"""Result model for promotion strategy execution outcomes."""

from dataclasses import dataclass

from ml.promotion.constants.constants import ProductionComparisonResult


@dataclass
class PromotionResult:
    """Normalized outcome payload returned by promotion strategies."""

    promotion_decision: bool
    beats_previous: bool
    previous_production_metrics: dict | None
    run_info: dict | None = None
    production_comparison: ProductionComparisonResult | None = None
