import logging
from pathlib import Path

from ml.config.validation_schemas.model_cfg import TrainModelConfig
from ml.runners.explainability.classes.classes import ExplainabilityMetrics
from ml.runners.explainability.explainers.base import Explainer
from ml.runners.explainability.explainers.tree_model.utils.adapter.get_adapter import \
    get_tree_model_adapter
from ml.runners.explainability.explainers.tree_model.utils.calculators.feature_importances import \
    get_feature_importances
from ml.runners.explainability.explainers.tree_model.utils.calculators.shap_importances import \
    get_shap_importances
from ml.runners.explainability.explainers.tree_model.utils.transformers.get_feature_names_and_transformed_X import \
    get_feature_names_and_transformed_X
from ml.utils.experiments.loading.pipeline import load_model_or_pipeline
from ml.utils.features.loading.X_and_y import load_X_and_y
from ml.utils.loaders import load_json

logger = logging.getLogger(__name__)

class ExplainTreeModel(Explainer):
    def explain(self, *, model_cfg: TrainModelConfig, train_dir: Path, top_k: int) -> tuple[ExplainabilityMetrics, list[dict]]:

        train_metadata_file = train_dir / "metadata.json"
        train_metadata = load_json(train_metadata_file)

        pipeline_file = Path(train_metadata.get("artifacts", {}).get("pipeline_path"))
        pipeline = load_model_or_pipeline(pipeline_file, "pipeline")

        X_test, _, feature_lineage = load_X_and_y(model_cfg, ["X_test", "y_test"], snapshot_selection=None, strict=True)

        feature_names, X_test_transformed = get_feature_names_and_transformed_X(pipeline, X_test)

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

        return explainability_metrics, feature_lineage