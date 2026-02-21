from dataclasses import dataclass
from typing import Optional
from ml.promotion.constants.constants import PreviousProductionRunIdentity, ThresholdComparisonResult
from ml.promotion.config.models import PromotionThresholds

@dataclass
class PromotionState:
    model_registry: dict
    archive_registry: dict
    evaluation_metrics: dict
    promotion_thresholds: PromotionThresholds
    current_prod_model_info: Optional[dict]
    previous_production_run_identity: PreviousProductionRunIdentity
    git_commit: str
    threshold_comparison: ThresholdComparisonResult
