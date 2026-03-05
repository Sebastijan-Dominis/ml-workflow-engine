"""Tree-model explainability runner implementation."""

import logging
from pathlib import Path

from ml.config.schemas.model_cfg import TrainModelConfig
from ml.features.loading.features_and_target import load_features_and_target
from ml.features.loading.resolve_feature_snapshots import resolve_feature_snapshots
from ml.features.splitting.splitting import get_splits
from ml.features.validation.validate_snapshot_ids import validate_snapshot_ids
from ml.metadata.validation.runners.training import validate_training_metadata
from ml.runners.explainability.constants.explainability_metrics_class import ExplainabilityMetrics
from ml.runners.explainability.constants.output import ExplainabilityOutput
from ml.runners.explainability.explainers.base import Explainer
from ml.runners.explainability.explainers.tree_model.utils.adapter.get_adapter import (
    get_tree_model_adapter,
)
from ml.runners.explainability.explainers.tree_model.utils.calculators.feature_importances import (
    get_feature_importances,
)
from ml.runners.explainability.explainers.tree_model.utils.calculators.shap_importances import (
    get_shap_importances,
)
from ml.runners.explainability.explainers.tree_model.utils.transformers.get_feature_names_and_transformed_features import (
    get_feature_names_and_transformed_features,
)
from ml.runners.shared.loading.get_snapshot_binding_from_training_metadata import (
    get_snapshot_binding_from_training_metadata,
)
from ml.runners.shared.loading.pipeline import load_model_or_pipeline
from ml.types import TabularSplits
from ml.utils.loaders import load_json

logger = logging.getLogger(__name__)

class ExplainTreeModel(Explainer):
    """Run explainability workflow for tree-based pipelines."""

    def explain(self, *, model_cfg: TrainModelConfig, train_dir: Path, top_k: int) -> ExplainabilityOutput:
        """Load artifacts/data and compute configured explainability outputs.

        Args:
            model_cfg: Validated training model configuration.
            train_dir: Directory containing training artifacts and metadata.
            top_k: Number of top features to include in explainability outputs.

        Returns:
            Explainability output with computed metrics and feature lineage.

        Raises:
            DataError: Propagated when required artifact/data lineage inputs are
                inconsistent or missing during explainability preparation.

        Notes:
            Explanations are computed on the test split after applying the same
            preprocessing pipeline used during training.

        Side Effects:
            Loads persisted training metadata/artifacts and may incur substantial
            compute for SHAP calculations.
        """

        splits: TabularSplits

        training_metadata_file = train_dir / "metadata.json"
        training_metadata_raw = load_json(training_metadata_file)
        training_metadata = validate_training_metadata(training_metadata_raw)

        if not training_metadata.artifacts.pipeline_path:
            msg = "Training metadata is missing the path to the trained pipeline artifact. Cannot proceed with explainability without the pipeline."
            logger.error(msg)
            raise ValueError(msg)
        pipeline_file = Path(training_metadata.artifacts.pipeline_path)
        pipeline = load_model_or_pipeline(pipeline_file, "pipeline")

        snapshot_binding = get_snapshot_binding_from_training_metadata(training_metadata)

        snapshot_selection = resolve_feature_snapshots(
            feature_store_path=Path(model_cfg.feature_store.path),
            feature_sets=model_cfg.feature_store.feature_sets,
            snapshot_binding=snapshot_binding
        )

        X, y, feature_lineage = load_features_and_target(model_cfg, snapshot_selection=snapshot_selection, strict=True)
        splits, splits_info = get_splits(
            X=X,
            y=y,
            split_cfg=model_cfg.split,
            data_type=model_cfg.data_type,
            task_cfg=model_cfg.task
        )

        validate_snapshot_ids(feature_lineage, snapshot_selection)

        X_test = splits.X_test

        feature_names, X_test_transformed = get_feature_names_and_transformed_features(pipeline, X_test)

        adapter = get_tree_model_adapter(pipeline[-1])

        top_k_feature_importances = get_feature_importances(
            feature_names=feature_names,
            pipeline=pipeline,
            model_cfg=model_cfg,
            top_k=top_k,
            adapter=adapter
        )

        top_k_shap_importances = get_shap_importances(
            feature_names=feature_names,
            model_configs=model_cfg,
            top_k=top_k,
            X_test_transformed=X_test_transformed,
            adapter=adapter
        )

        explainability_metrics = ExplainabilityMetrics(
            top_k_feature_importances=top_k_feature_importances,
            top_k_shap_importances=top_k_shap_importances,
        )

        output = ExplainabilityOutput(
            explainability_metrics=explainability_metrics,
            feature_lineage=feature_lineage
        )

        return output
