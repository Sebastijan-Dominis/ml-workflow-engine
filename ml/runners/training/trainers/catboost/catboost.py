"""Binary classification training utilities using CatBoost.

This module contains helper routines and a top-level training function
used to train binary classification models with CatBoost within the
project training framework. It provides deterministic data loading,
dynamic import of model-specific pipeline components from the
``ml.components`` package, pipeline construction, model definition,
and training orchestration.

The public entrypoint is ``train_binary_classification_with_catboost`` which
returns a fitted ``sklearn.pipeline.Pipeline`` combining preprocessing
steps and the trained CatBoost model.
"""

from pathlib import Path

from ml.config.hashing import compute_config_hash
from ml.config.validation_schemas.model_cfg import TrainModelConfig
from ml.runners.training.constants.output import TRAIN_OUTPUT
from ml.runners.training.trainers.base import Trainer
from ml.runners.training.trainers.catboost.train_catboost_model import \
    train_catboost_model
from ml.runners.training.utils.metrics.compute_metrics import compute_metrics
from ml.runners.training.utils.model_specific.catboost import prepare_model
from ml.utils.catboost.build_pipeline_with_model import \
    build_pipeline_with_model
from ml.utils.experiments.class_weights.models import DataStats
from ml.utils.experiments.class_weights.resolve_class_weighting import \
    resolve_class_weighting
from ml.utils.experiments.class_weights.stats_resolver import \
    compute_data_stats
from ml.utils.features.cat_features import get_cat_features
from ml.utils.features.loading.schemas import load_schemas
from ml.utils.features.loading.X_and_y import load_X_and_y
from ml.utils.features.splitting.splitting import get_splits
from ml.utils.features.validation.validate_contract import \
    validate_model_feature_pipeline_contract
from ml.utils.loaders import load_yaml
from ml.utils.features.transform_target import transform_target


class TrainCatboost(Trainer):
    def train(
        self, 
        model_cfg: TrainModelConfig, 
        *,
        strict: bool,
        failure_management_dir: Path
    ) -> TRAIN_OUTPUT:
        stats: DataStats

        X, y, lineage = load_X_and_y(model_cfg, snapshot_selection=None, strict=strict)
        splits, splits_info = get_splits(
            X=X,
            y=y,
            split_cfg=model_cfg.split,
            data_type=model_cfg.data_type,
            task_cfg=model_cfg.task
        )

        X_train = splits.X_train
        y_train = transform_target(
            splits.y_train, 
            transform_config=model_cfg.target.transform,
            split_name="train"
        )
        X_val = splits.X_val
        y_val = transform_target(
            splits.y_val, 
            transform_config=model_cfg.target.transform,
            split_name="val"
        )

        input_schema, derived_schema = load_schemas(model_cfg)

        cat_features = get_cat_features(model_cfg, input_schema, derived_schema)

        pipeline_path = Path(f"{model_cfg.pipeline.path}").resolve()
        pipeline_cfg = load_yaml(pipeline_path)
        pipeline_cfg_hash = compute_config_hash(pipeline_cfg)

        cat_features = get_cat_features(model_cfg, input_schema, derived_schema)

        validate_model_feature_pipeline_contract(
            model_cfg,
            pipeline_cfg,
            cat_features
        )

        class_weights = {}
        if model_cfg.task.type == "classification":
            stats = compute_data_stats(y_train)
            class_weights = resolve_class_weighting(model_cfg, stats, "catboost")

        model = prepare_model(
            model_cfg, 
            cat_features=cat_features, 
            class_weights=class_weights,
            failure_management_dir=failure_management_dir
        )

        pipeline = build_pipeline_with_model(
            model_cfg=model_cfg,
            pipeline_cfg=pipeline_cfg,
            input_schema=input_schema,
            derived_schema=derived_schema,
            model=model
        )

        model_trained, pipeline_trained = train_catboost_model(
            model_cfg, 
            steps=pipeline.steps, 
            X_train=X_train, 
            y_train=y_train, 
            X_val=X_val, 
            y_val=y_val
        )

        metrics = compute_metrics(
            model=model_trained,
            pipeline=pipeline_trained,
            model_cfg=model_cfg,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val
        )

        output = TRAIN_OUTPUT(
            model=model_trained,
            pipeline=pipeline_trained,
            lineage=lineage,
            metrics=metrics,
            pipeline_cfg_hash=pipeline_cfg_hash
        )

        return output