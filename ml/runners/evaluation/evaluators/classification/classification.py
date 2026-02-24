import logging
from pathlib import Path

import pandas as pd

from ml.config.validation_schemas.model_cfg import TrainModelConfig
from ml.runners.evaluation.constants.data_splits import DataSplits
from ml.runners.evaluation.constants.output import EVALUATE_OUTPUT
from ml.runners.evaluation.evaluators.base import Evaluator
from ml.runners.evaluation.evaluators.classification.metrics import \
    evaluate_model
from ml.utils.experiments.loading.pipeline import load_model_or_pipeline
from ml.utils.features.loading.X_and_y import load_X_and_y
from ml.utils.features.splitting.splitting import get_splits
from ml.utils.loaders import load_json

logger = logging.getLogger(__name__)

class EvaluateClassification(Evaluator):
    def evaluate(self, *, model_cfg: TrainModelConfig, strict: bool, best_threshold: float | None, train_dir: Path) -> EVALUATE_OUTPUT:
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
            raise ValueError(msg)

        # Get data splits
        X, y, lineage = load_X_and_y(
            model_cfg, 
            snapshot_selection=None, 
            drop_row_id=False, 
            strict=strict
        )
        splits = get_splits(
            X=X,
            y=y,
            split_cfg=model_cfg.split,
            data_type=model_cfg.data_type
        )
        X_train = splits.X_train
        y_train = splits.y_train
        X_val = splits.X_val
        y_val = splits.y_val
        X_test = splits.X_test
        y_test = splits.y_test

        # TODO: Import row_id, validate, and merge with X for each split to ensure row_id is available for creating prediction DataFrames and consistent with original data
        data_splits = DataSplits(
            train=(X_train, y_train),
            val=(X_val, y_val),
            test=(X_test, y_test)
        )

        # Evaluate the model
        metrics, prediction_dfs = evaluate_model(model_cfg, pipeline=pipeline, data_splits=data_splits, best_threshold=best_threshold)

        output = EVALUATE_OUTPUT(
            metrics=metrics,
            prediction_dfs=prediction_dfs,
            lineage=lineage
        )

        return output