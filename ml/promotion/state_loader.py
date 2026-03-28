"""State-loading service for promotion workflow execution."""

import logging

from ml.promotion.comparisons.thresholds import compare_against_thresholds
from ml.promotion.constants.constants import PreviousProductionRunIdentity
from ml.promotion.context import PromotionContext
from ml.promotion.getters.get import extract_thresholds
from ml.promotion.state import PromotionState
from ml.promotion.validation.promotion_thresholds import validate_promotion_thresholds
from ml.utils.git import get_git_commit
from ml.utils.loaders import load_json, load_yaml

logger = logging.getLogger(__name__)

class PromotionStateLoader:
    """Loads registries, thresholds, and metrics into a promotion state object."""

    def load(self, context: PromotionContext) -> PromotionState:
        """Load and assemble promotion state from persisted artifacts.

        Args:
            context: Promotion context containing paths and run arguments.

        Returns:
            Promotion state with registries, thresholds, metrics, and comparisons.
        """
        model_registry = load_yaml(context.paths.registry_path)
        archive_registry = load_yaml(context.paths.archive_path)

        global_thresholds = load_yaml(
            context.paths.promotion_configs_dir / "thresholds.yaml"
        )

        evaluation_metrics_file = load_json(
            context.paths.eval_run_dir / "metrics.json"
        )
        evaluation_metrics = evaluation_metrics_file.get("metrics", {})

        promotion_thresholds_raw = extract_thresholds(
            promotion_thresholds=global_thresholds,
            problem=context.args.problem,
            segment=context.args.segment,
        )

        promotion_thresholds = validate_promotion_thresholds(
            promotion_thresholds_raw
        )

        current_prod_model_info = (
            model_registry
            .get(context.args.problem, {})
            .get(context.args.segment, {})
            .get("production")
        )

        git_commit = get_git_commit()

        previous_identity = PreviousProductionRunIdentity(
            experiment_id=current_prod_model_info.get("experiment_id") if current_prod_model_info else None,
            train_run_id=current_prod_model_info.get("train_run_id") if current_prod_model_info else None,
            eval_run_id=current_prod_model_info.get("eval_run_id") if current_prod_model_info else None,
            explain_run_id=current_prod_model_info.get("explain_run_id") if current_prod_model_info else None,
            promotion_id=current_prod_model_info.get("promotion_id") if current_prod_model_info else None
        )

        threshold_comparison = compare_against_thresholds(
            evaluation_metrics=evaluation_metrics,
            promotion_thresholds=promotion_thresholds,
        )

        promotion_state = PromotionState(
            model_registry=model_registry,
            archive_registry=archive_registry,
            evaluation_metrics=evaluation_metrics,
            promotion_thresholds=promotion_thresholds,
            current_prod_model_info=current_prod_model_info,
            previous_production_run_identity=previous_identity,
            git_commit=git_commit,
            threshold_comparison=threshold_comparison
        )

        return promotion_state
