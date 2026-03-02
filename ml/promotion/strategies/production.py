"""Production-stage promotion strategy implementation."""

import logging

from ml.exceptions import RuntimeMLException
from ml.promotion.comparisons.production import \
    compare_against_production_model
from ml.promotion.constants.constants import RunnersMetadata
from ml.promotion.persistence.prepare import prepare_run_information
from ml.promotion.result import PromotionResult
from ml.promotion.strategies.base import PromotionStrategy

logger = logging.getLogger(__name__)

class ProductionPromotionStrategy(PromotionStrategy):
    """Apply thresholds and production-baseline checks for promotion."""

    def execute(self, context, state):
        """Execute production promotion decision logic.

        Args:
            context: Promotion runtime context with arguments, paths, and metadata.
            state: Loaded promotion state with thresholds and production metrics.

        Returns:
            Promotion decision result for the production strategy.

        Raises:
            RuntimeMLException: If required runner metadata is missing from
                promotion context.

        Notes:
            Promotion requires both threshold compliance and improvement against
            currently deployed production metrics.

        Side Effects:
            Emits promotion decision logs and may construct promotion run metadata
            payload for downstream persistence.
        """
        if not isinstance(context.runners_metadata, RunnersMetadata):
            msg = "Runners metadata is required for production promotion strategy but was not found in context."
            logger.error(msg)
            raise RuntimeMLException(msg)

        threshold_comparison = state.threshold_comparison

        run_info = None

        production_comparison = compare_against_production_model(
            evaluation_metrics=state.evaluation_metrics,
            current_prod_model_info=state.current_prod_model_info,
            metric_sets=threshold_comparison.target_sets,
            metric_names=threshold_comparison.target_metrics,
            directions=threshold_comparison.directions,
        )

        promotion_decision = (
            threshold_comparison.meets_thresholds
            and production_comparison.beats_previous
        )

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

            reason = "Model meets all promotion criteria."

            previous_id = (
                state.current_prod_model_info.get("promotion_id")
                if state.current_prod_model_info
                else None
            )

            logger.info(
                "Model promoted and previous production model with promotion_id '%s' archived successfully.", previous_id
            )
        else:
            reasons = []
            if not threshold_comparison.meets_thresholds:
                reasons.append(threshold_comparison.message)
            if not production_comparison.beats_previous:
                reasons.append(production_comparison.message)
            reason = "; ".join(reasons)
            logger.info(f"Model promotion criteria not met. Reasoning: {reason}")

        return PromotionResult(
            promotion_decision=promotion_decision,
            beats_previous=production_comparison.beats_previous,
            run_info=run_info,
            previous_production_metrics=production_comparison.previous_production_metrics,
            production_comparison=production_comparison
        )