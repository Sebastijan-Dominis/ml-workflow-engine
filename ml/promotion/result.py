from dataclasses import dataclass
from typing import Optional

from ml.promotion.constants.constants import ProductionComparisonResult

@dataclass
class PromotionResult:
    promotion_decision: bool
    beats_previous: bool
    previous_production_metrics: Optional[dict]
    run_info: Optional[dict] = None
    production_comparison: Optional[ProductionComparisonResult] = None