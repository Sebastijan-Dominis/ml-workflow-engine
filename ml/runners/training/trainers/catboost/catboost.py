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

from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.pipeline import Pipeline

from ml.runners.training.trainers.base import Trainer
from ml.config.validation_schemas.model_cfg import TrainModelConfig
from ml.runners.training.trainers.catboost.train_catboost_model import train_catboost_model
from ml.runners.training.utils.metrics.compute_metrics import compute_metrics
from ml.runners.training.utils.model_specific.catboost import prepare_model
from ml.utils.catboost.build_pipeline_with_model import build_pipeline_with_model
from ml.utils.features.cat_features import get_cat_features
from ml.utils.features.loading.resolve_feature_snapshots import resolve_feature_snapshots
from ml.utils.features.loading.schemas import load_schemas
from ml.utils.features.loading.X_and_y import load_X_and_y
from ml.utils.loaders import load_yaml
from ml.config.hashing import compute_config_hash
from ml.utils.features.validation import validate_model_feature_pipeline_contract


class TrainCatboost(Trainer):
    def train(self, model_cfg: TrainModelConfig, strict: bool) -> tuple[CatBoostClassifier | CatBoostRegressor, Pipeline, list[dict], dict[str, float], str | None]:
        """Train a binary classification model using CatBoost and project components.

        This is the high-level routine used by the training CLI to execute a
        complete training run. It performs data loading, dynamic component
        import for model-specific preprocessing, model construction, pipeline
        assembly, training, and returns a fitted pipeline object.

    logger = logging.getLogger(__name__)
        Args:
            name_version (str): Component module name under ``ml.components``.
            cfg (dict): Validated configuration dictionary.

        Returns:
            tuple[CatBoostClassifier | CatBoostRegressor, Pipeline, list[dict], dict[str, float], str | None]: A fitted sklearn Pipeline containing preprocessing and
            the trained CatBoost model, along with the pipeline hash.

        Raises:
            Exception: Any exception during the training process is logged and
            re-raised to signal a fatal training error.
        """

        snapshot_selection = resolve_feature_snapshots(Path(model_cfg.feature_store.path), model_cfg.feature_store.feature_sets)
        X_train, y_train, lineage_train = load_X_and_y(model_cfg, ["X_train", "y_train"], snapshot_selection=snapshot_selection, strict=strict)
        X_val, y_val, _ = load_X_and_y(model_cfg, ["X_val", "y_val"], snapshot_selection=snapshot_selection, strict=strict)

        input_schema, derived_schema = load_schemas(model_cfg)

        cat_features = get_cat_features(input_schema, derived_schema)

        pipeline_path = Path(f"{model_cfg.pipeline.path}").resolve()
        pipeline_cfg = load_yaml(pipeline_path)
        pipeline_hash = compute_config_hash(pipeline_cfg)

        cat_features = get_cat_features(input_schema, derived_schema)

        validate_model_feature_pipeline_contract(
            model_cfg,
            pipeline_cfg,
            cat_features
        )

        model = prepare_model(model_cfg, cat_features)

        pipeline = build_pipeline_with_model(
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



        return model_trained, pipeline_trained, lineage_train, metrics, pipeline_hash