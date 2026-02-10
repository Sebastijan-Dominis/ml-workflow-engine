import logging
from pathlib import Path

from ml.runners.evaluation.evaluators.base import Evaluator
from ml.config.validation_schemas.model_cfg import TrainModelConfig
from ml.utils.loaders import load_json
from ml.utils.features.loading.resolve_feature_snapshots import resolve_feature_snapshots
from ml.utils.features.loading.X_and_y import load_X_and_y
from ml.utils.features.loading.pipeline import load_pipeline
from ml.utils.experiments.transform_y_to_series import transform_y_to_series
from ml.runners.evaluation.evaluators.classification.metrics import evaluate_model

logger = logging.getLogger(__name__)

class EvaluateClassification(Evaluator):
    def evaluate(self, *, model_cfg: TrainModelConfig, strict: bool, best_threshold: float | None, train_dir: Path) -> tuple[dict[str, dict[str, float]], list[dict]]:
        if best_threshold is None and model_cfg.task.subtype and model_cfg.task.subtype.lower() == "binary":
            msg = f"Best threshold for classification evaluation is not defined for task '{model_cfg.task.type}' with subtype '{model_cfg.task.subtype}'. Defaulting to 0.5."
            logger.warning(msg)
            best_threshold = 0.5

        # Load the trained pipeline from the training artifacts
        metadata_file = train_dir / "metadata.json"
        metadata = load_json(metadata_file)

        pipeline_file = metadata["artifacts"]["pipeline_path"]
        pipeline = load_pipeline(pipeline_file)

        # Get data splits
        snapshot_selection = resolve_feature_snapshots(Path(model_cfg.feature_store.path), model_cfg.feature_store.feature_sets)
        X_train, y_train, lineage_train = load_X_and_y(model_cfg, ["X_train", "y_train"], snapshot_selection=snapshot_selection, strict=strict)
        X_val, y_val, _ = load_X_and_y(model_cfg, ["X_val", "y_val"], snapshot_selection=snapshot_selection, strict=strict)
        X_test, y_test, _ = load_X_and_y(model_cfg, ["X_test", "y_test"], snapshot_selection=snapshot_selection, strict=strict)

        # Transform y to Series if it's a DataFrame for metric compatibility
        y_train = transform_y_to_series(y_train)
        y_val = transform_y_to_series(y_val)
        y_test = transform_y_to_series(y_test)

        data_splits = {
            "train": (X_train, y_train),
            "val": (X_val, y_val),
            "test": (X_test, y_test)
        }

        # Evaluate the model
        evaluation_results = evaluate_model(model_cfg, pipeline=pipeline, data_splits=data_splits, best_threshold=best_threshold)

        # Return evaluation results, along with feature lineage
        return evaluation_results, lineage_train