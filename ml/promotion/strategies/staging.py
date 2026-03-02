"""Staging-stage promotion strategy implementation."""

import logging

from ml.exceptions import RuntimeMLException
from ml.promotion.constants.constants import RunnersMetadata
from ml.promotion.persistence.prepare import prepare_run_information
from ml.promotion.result import PromotionResult
from ml.promotion.strategies.base import PromotionStrategy

logger = logging.getLogger(__name__)

class StagingPromotionStrategy(PromotionStrategy):
    """Apply threshold-only checks for staging promotion decisions."""

    def execute(self, context, state):
        """Execute staging promotion decision logic.

        Args:
            context: Promotion runtime context with arguments, paths, and metadata.
            state: Loaded promotion state with thresholds and metrics.

        Returns:
            Promotion decision result for the staging strategy.
        """
        if not isinstance(context.runners_metadata, RunnersMetadata):
            msg = "Runners metadata is required for staging promotion strategy but was not found in context."
            logger.error(msg)
            raise RuntimeMLException(msg)

        promotion_decision = state.threshold_comparison.meets_thresholds

        run_info = None

        if promotion_decision:
            run_info = prepare_run_information(
                args=context.args,
                experiment_id=context.args.experiment_id,
                train_run_id=context.args.train_run_id,
                eval_run_id=context.args.eval_run_id,
                explain_run_id=context.args.explain_run_id,
                run_id=context.run_id,
                timestamp=context.timestamp,
                explain_metadata=context.runners_metadata.explain_metadata,
                training_metadata=context.runners_metadata.train_metadata,
                metrics=state.evaluation_metrics,
                git_commit=state.git_commit
            )

            reason = "Model beats the thresholds. No comparison against production model for staging promotion."

            logger.info("Model promoted to staging successfully.")
        else:
            reason = state.threshold_comparison.message
            logger.info(f"Model staging criteria not met. Reasoning: {reason}")

        return PromotionResult(
            promotion_decision=promotion_decision,
            beats_previous=False,
            run_info=run_info,
            previous_production_metrics=None,
            production_comparison=None
        )