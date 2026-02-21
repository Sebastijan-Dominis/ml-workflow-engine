import logging

from ml.exceptions import RuntimeMLException
from ml.promotion.context import PromotionContext
from ml.promotion.persistence.prepare import prepare_metadata
from ml.promotion.persistence.registry import (persist_registry_diff,
                                               update_registry_and_archive)
from ml.promotion.result import PromotionResult
from ml.promotion.state import PromotionState
from ml.utils.persistence.save_metadata import save_metadata

logger = logging.getLogger(__name__)

class PromotionPersister:

    def _build_reason_and_log_msg(self, context: PromotionContext, state: PromotionState, result: PromotionResult) -> tuple[str, str]:
        if result.promotion_decision:
            if context.args.stage == "production":
                reason = "Model meets all promotion criteria."
                previous_id = (
                    state.current_prod_model_info.get("promotion_id")
                    if state.current_prod_model_info
                    else None
                )
                log_msg = f"Model promoted and previous production model with promotion_id {previous_id} archived successfully."
                return reason, log_msg
            
            elif context.args.stage == "staging":
                reason = "Model beats the thresholds. No comparison against production model for staging promotion."
                log_msg = "Model promoted to staging successfully."
                return reason, log_msg
            
            else:
                msg = f"Invalid stage '{context.args.stage}' in promotion context. Expected 'production' or 'staging'."
                logger.error(msg)
                raise RuntimeMLException(msg)
            
        else:
            if context.args.stage == "production":
                threshold_comparison = state.threshold_comparison

                if not result.production_comparison:
                    msg = "Production comparison result is missing in the promotion result. This should not happen for production stage."
                    logger.error(msg)
                    raise RuntimeMLException(msg)

                reasons = []
                if not threshold_comparison.meets_thresholds:
                    reasons.append(threshold_comparison.message)
                if not result.production_comparison.beats_previous:
                    reasons.append(result.production_comparison.message)
                reason = "; ".join(reasons)
                log_msg = f"Model promotion criteria not met. Reasoning: {reason}"
                return reason, log_msg

            elif context.args.stage == "staging":
                reason = state.threshold_comparison.message
                log_msg = f"Model staging criteria not met. Reasoning: {reason}"
                return reason, log_msg
            
            else:
                msg = f"Invalid stage '{context.args.stage}' in promotion context. Expected 'production' or 'staging'."
                logger.error(msg)
                raise RuntimeMLException(msg)

    def persist(
        self,
        context: PromotionContext,
        state: PromotionState,
        result: PromotionResult,
    ):
        updated_registry = None
        reason, log_msg = self._build_reason_and_log_msg(context, state, result)

        if result.promotion_decision:
            if not result.run_info:
                msg = "Promotion decision is True but run_info is missing in the result. This should not happen."
                logger.error(msg)
                raise RuntimeMLException(msg)

            updated_registry = update_registry_and_archive(
                model_registry=state.model_registry,
                archive_registry=state.archive_registry,
                run_info=result.run_info,
                stage=context.args.stage,
                problem=context.args.problem,
                segment=context.args.segment,
                registry_path=context.paths.registry_path,
                archive_path=context.paths.archive_path,
            )

        logger.info(log_msg)

        if updated_registry is not None:
            persist_registry_diff(
                previous_registry=state.model_registry,
                updated_registry=updated_registry,
                run_dir=context.paths.run_dir,
            )

        metadata = prepare_metadata(
            run_id=context.run_id,
            args=context.args,
            metrics=state.evaluation_metrics,
            previous_production_metrics=result.previous_production_metrics,
            promotion_thresholds=state.promotion_thresholds,
            promoted=result.promotion_decision,
            beats_previous=result.beats_previous,
            reason=reason,
            git_commit=state.git_commit,
            timestamp=context.timestamp,
            previous_production_run_identity=state.previous_production_run_identity,
            train_run_dir=context.paths.train_run_dir,
        )

        save_metadata(
            metadata = metadata,
            target_dir = context.paths.run_dir
        )