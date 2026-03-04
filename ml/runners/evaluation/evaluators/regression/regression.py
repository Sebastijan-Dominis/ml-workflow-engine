"""Regression evaluator implementation for trained model artifacts."""

import logging
from pathlib import Path

from ml.config.schemas.model_cfg import TrainModelConfig
from ml.exceptions import PipelineContractError
from ml.features.loading.features_and_target import load_features_and_target
from ml.features.loading.resolve_feature_snapshots import resolve_feature_snapshots
from ml.features.splitting.splitting import get_splits
from ml.features.validation.validate_snapshot_ids import validate_snapshot_ids
from ml.metadata.validation.runners.training import validate_training_metadata
from ml.modeling.models.feature_lineage import FeatureLineage
from ml.runners.evaluation.constants.data_splits import DataSplits
from ml.runners.evaluation.constants.output import EvaluateOutput
from ml.runners.evaluation.evaluators.base import Evaluator
from ml.runners.evaluation.evaluators.regression.metrics import evaluate_model
from ml.runners.evaluation.models.predictions import PredictionArtifacts
from ml.runners.shared.loading.get_snapshot_binding_from_training_metadata import (
    get_snapshot_binding_from_training_metadata,
)
from ml.runners.shared.loading.pipeline import load_model_or_pipeline
from ml.utils.loaders import load_json

logger = logging.getLogger(__name__)


class EvaluateRegression(Evaluator):
    """Evaluate regression models across train/val/test splits."""

    def evaluate(
        self,
        *,
        model_cfg: TrainModelConfig,
        strict: bool,
        best_threshold: float | None,  # unused, kept for interface consistency
        train_dir: Path
    ) -> EvaluateOutput:
        """Load artifacts and data, run split-wise regression evaluation.

        Args:
            model_cfg: Validated training model configuration.
            strict: Whether data-loading checks should fail strictly.
            best_threshold: Unused threshold parameter kept for interface parity.
            train_dir: Directory containing training artifacts and metadata.

        Returns:
            Evaluation output with metrics, predictions, and feature lineage.

        Raises:
            PipelineContractError: If loaded pipeline cannot produce predictions.

        Notes:
            Uses training metadata lineage bindings to resolve feature snapshots so
            evaluation inputs match training-time feature versions.

        Side Effects:
            Loads persisted artifacts and executes split-wise inference on
            train/validation/test data.
        """
        prediction_dfs: PredictionArtifacts
        feature_lineage: list[FeatureLineage]

        # -------------------------
        # Load trained pipeline
        # -------------------------
        training_metadata_file = train_dir / "metadata.json"
        training_metadata_raw = load_json(training_metadata_file)
        training_metadata = validate_training_metadata(training_metadata_raw)

        if not training_metadata.artifacts.pipeline_path:
            msg = "Training metadata is missing the path to the trained pipeline artifact. Cannot proceed with evaluation without the pipeline."
            logger.error(msg)
            raise PipelineContractError(msg)

        pipeline_file = Path(training_metadata.artifacts.pipeline_path)
        pipeline = load_model_or_pipeline(pipeline_file, "pipeline")

        if not hasattr(pipeline, "predict"):
            msg = "Loaded pipeline does not implement 'predict', required for regression evaluation."
            logger.error(msg)
            raise PipelineContractError(msg)

        # -------------------------
        # Load data
        # -------------------------
        snapshot_binding = get_snapshot_binding_from_training_metadata(training_metadata)

        snapshot_selection = resolve_feature_snapshots(
            feature_store_path=Path(model_cfg.feature_store.path),
            feature_sets=model_cfg.feature_store.feature_sets,
            snapshot_binding=snapshot_binding,
        )

        X, y, feature_lineage = load_features_and_target(
            model_cfg,
            snapshot_selection=snapshot_selection,
            drop_row_id=False,
            strict=strict,
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

        # -------------------------
        # Evaluate
        # -------------------------
        metrics, prediction_dfs = evaluate_model(
            pipeline=pipeline,
            data_splits=data_splits,
            transform_cfg=model_cfg.target.transform,
        )

        output = EvaluateOutput(**{
            "metrics": metrics,
            "prediction_dfs": prediction_dfs,
            "lineage": feature_lineage,
        })

        return output
