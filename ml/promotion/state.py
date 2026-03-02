"""State model containing promotion inputs loaded from persisted artifacts."""

from dataclasses import dataclass
from typing import Optional

from ml.promotion.config.models import PromotionThresholds
from ml.promotion.constants.constants import (PreviousProductionRunIdentity,
                                              ThresholdComparisonResult)


@dataclass
class PromotionState:
    """Immutable-like snapshot of loaded state for promotion decision flow."""

    model_registry: dict
    archive_registry: dict
    evaluation_metrics: dict
    promotion_thresholds: PromotionThresholds
    current_prod_model_info: Optional[dict]
    previous_production_run_identity: PreviousProductionRunIdentity
    git_commit: str
    threshold_comparison: ThresholdComparisonResult
