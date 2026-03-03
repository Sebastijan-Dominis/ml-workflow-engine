"""Classification evaluator implementation for trained model artifacts."""

import logging
from pathlib import Path

from ml.config.schemas.model_cfg import TrainModelConfig
from ml.exceptions import PipelineContractError
from ml.runners.evaluation.constants.data_splits import DataSplits
from ml.runners.evaluation.constants.output import EvaluateOutput
from ml.runners.evaluation.evaluators.base import Evaluator
from ml.runners.evaluation.evaluators.classification.metrics import \
    evaluate_model
from ml.utils.experiments.loading.get_snapshot_binding_from_training_metadata import \
    get_snapshot_binding_from_training_metadata
from ml.utils.experiments.loading.pipeline import load_model_or_pipeline
from ml.utils.features.loading.resolve_feature_snapshots import \
    resolve_feature_snapshots
from ml.utils.features.loading.X_and_y import load_X_and_y
from ml.utils.features.splitting.splitting import get_splits
from ml.utils.features.validation.validate_snapshot_ids import \
    validate_snapshot_ids
from ml.utils.loaders import load_json

logger = logging.getLogger(__name__)

class ClassificationEvaluator(Evaluator):
    """Evaluate binary classification models across train/val/test splits."""

    def evaluate(self, *, model_cfg: TrainModelConfig, strict: bool, best_threshold: float | None, train_dir: Path) -> EvaluateOutput:
        """Load artifacts and data, run split-wise evaluation, return outputs.

        Args:
            model_cfg: Validated training model configuration.
            strict: Whether data-loading checks should fail strictly.
            best_threshold: Optional decision threshold for binary classification.
            train_dir: Directory containing training artifacts and metadata.

        Returns:
            Evaluation output with metrics, predictions, and feature lineage.

        Raises:
            PipelineContractError: If the loaded pipeline does not expose
                ``predict_proba`` required for classification evaluation.

        Notes:
            Uses training metadata lineage bindings to resolve feature snapshots so
            evaluation inputs match training-time feature versions.

        Side Effects:
            Loads persisted artifacts and executes split-wise inference on
            train/validation/test data.
        """

        data_splits: DataSplits

        if best_threshold is None and model_cfg.task.subtype and model_cfg.task.subtype.lower() == "binary":
            msg = f"Best threshold for classification evaluation is not defined for task '{model_cfg.task.type}' with subtype '{model_cfg.task.subtype}'. Defaulting to 0.5."
            logger.warning(msg)
            best_threshold = 0.5

        # Load the trained pipeline from the training artifacts
        train_metadata_file = train_dir / "metadata.json"
        train_metadata = load_json(train_metadata_file)

        pipeline_file = Path(train_metadata.get("artifacts", {}).get("pipeline_path"))
        pipeline = load_model_or_pipeline(pipeline_file, "pipeline")

        if not hasattr(pipeline, "predict_proba"):
            msg = f"The loaded pipeline does not implement 'predict_proba', which is required for classification evaluation. Please ensure the pipeline is a probabilistic classifier."
            logger.error(msg)
            raise PipelineContractError(msg)

        # Get data splits
        snapshot_binding = get_snapshot_binding_from_training_metadata(train_metadata)
        
        snapshot_selection = resolve_feature_snapshots(
            feature_store_path=Path(model_cfg.feature_store.path),
            feature_sets=model_cfg.feature_store.feature_sets,
            snapshot_binding=snapshot_binding
        )
        X, y, feature_lineage = load_X_and_y(
            model_cfg, 
            snapshot_selection=snapshot_selection, 
            drop_row_id=False, 
            strict=strict
        )

        validate_snapshot_ids(feature_lineage, snapshot_selection)

        splits, splits_info = get_splits(
            X=X,
            y=y,
            split_cfg=model_cfg.split,
            data_type=model_cfg.data_type,
            task_cfg=model_cfg.task
        )
        X_train = splits.X_train
        y_train = splits.y_train
        X_val = splits.X_val
        y_val = splits.y_val
        X_test = splits.X_test
        y_test = splits.y_test

        data_splits = DataSplits(
            train=(X_train, y_train),
            val=(X_val, y_val),
            test=(X_test, y_test)
        )

        # Evaluate the model
        metrics, prediction_dfs = evaluate_model(
            model_cfg, 
            pipeline=pipeline, 
            data_splits=data_splits, 
            best_threshold=best_threshold
        )

        output = EvaluateOutput(
            metrics=metrics,
            prediction_dfs=prediction_dfs,
            lineage=feature_lineage
        )

        return output