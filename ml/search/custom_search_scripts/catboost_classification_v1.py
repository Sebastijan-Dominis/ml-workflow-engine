import logging
logger = logging.getLogger(__name__)
import importlib
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline

from ml.search.utils import get_data, search_best_params

param_distributions_1 = {
    'model__depth': [4, 6, 8, 10],
    'model__learning_rate': [0.005, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3],
    'model__l2_leaf_reg': [1, 3, 5, 7, 9],
    'model__bagging_temperature': [0, 0.25, 0.5, 0.75, 1, 2, 5],
    'model__border_count': [32, 64, 128, 254],
    'model__min_data_in_leaf': [1, 3, 5, 8, 10, 15, 20],
    # 'model__colsample_bylevel': [0.6, 0.8, 1.0], # Ok to explore when using CPU
    'model__colsample_bylevel': [1.0], # Set to 1.0 to avoid issues with GPU training
    'model__random_strength': [1, 2, 5, 8, 10, 12]
}

def import_components(cfg):
    components_path = cfg["artifacts"]["components_path"]
    try:
        return importlib.import_module(components_path.replace('.py', '').replace('/', '.'))
    except Exception:
        logger.exception(f"Failed to import components from {components_path}.")
        raise

def get_feature_names_from_schema(cfg):
    schema_path = cfg["features"]["schema_path"]
    try:
        schema = pd.read_csv(schema_path)

        engineered_categorical_features = cfg["features"].get("engineered_categorical_features", [])
        categorical_features = schema.loc[schema["dtype"].isin(['object', 'category', 'string']), "feature"].tolist()
        numerical_features = schema.loc[schema["dtype"].isin(['int64', 'float64', 'int32', 'float32', 'int16', 'int8']), "feature"].tolist()
        required_features = categorical_features + numerical_features
        cat_features = categorical_features + engineered_categorical_features

        return categorical_features, required_features, cat_features
    except Exception:
        logger.exception(f"Failed to extract feature names from {schema_path}.")
        raise

def create_pipeline(cfg, search_phase):
    components_module = import_components(cfg)
    categorical_features, required_features, cat_features = get_feature_names_from_schema(cfg)

    try:
        schema_validator = components_module.SchemaValidator(required_features=required_features)
        fill_categorical_missing = components_module.FillCategoricalMissing(categorical_features=categorical_features)
        feature_engineer = components_module.FeatureEngineer(engineered_features=cfg["features"].get("engineered_features", []))
        feature_selector = components_module.FeatureSelector(selected_features=required_features + cfg["features"].get("engineered_features", []))
    except Exception:
        logger.exception("Failed to initialize preprocessing pipeline components.")
        raise

    try:
        model = CatBoostClassifier(
            # Basic hyperparameters
            iterations=cfg["search"][search_phase]["iterations"],
            task_type='GPU',             # enable GPU
            devices='0',                 # GPU device
            verbose=100,                
            random_state=42,
            cat_features=cat_features,
            class_weights=cfg.get("class_weights", None)
        )
    except Exception:
        logger.exception("Failed to initialize CatBoostClassifier.")
        raise

    steps = [
        ("schema_validator", schema_validator),
        ("fill_categorical_missing", fill_categorical_missing),
        ("feature_engineer", feature_engineer),
        ("feature_selector", feature_selector),
        ("model", model),
    ]

    try:
        pipeline = Pipeline(steps)
    except Exception:
        logger.exception("Failed to create the pipeline.")
        raise

    return pipeline

# Helper function
def refine_int(center, offsets, low, high):
    values = {center}
    for o in offsets:
        values.add(center + o)
        values.add(center - o)
    return sorted(v for v in values if low <= v <= high)

# Helper function
def refine_float_mult(center, factors, low, high, decimals=5):
    values = set()
    for f in factors:
        v = center * f
        if low <= v <= high:
            values.add(round(v, decimals))
    values.add(round(center, decimals))
    return sorted(values)

def prepare_narrow_search_params(best_params):
    narrow_params = {}

    # Tree depth
    if "model__depth" in best_params:
        narrow_params["model__depth"] = refine_int(
            best_params["model__depth"],
            offsets=[1, 2],
            low=2,
            high=12
        )

    # Learning rate
    if "model__learning_rate" in best_params:
        narrow_params["model__learning_rate"] = refine_float_mult(
            best_params["model__learning_rate"],
            factors=[0.7, 0.85, 1.0, 1.15, 1.3],
            low=0.003,
            high=0.5
        )

    # L2 regularization
    if "model__l2_leaf_reg" in best_params:
        narrow_params["model__l2_leaf_reg"] = refine_int(
            best_params["model__l2_leaf_reg"],
            offsets=[1, 2, 3],
            low=1,
            high=30
        )

    # Bagging temperature
    if "model__bagging_temperature" in best_params:
        narrow_params["model__bagging_temperature"] = refine_float_mult(
            best_params["model__bagging_temperature"],
            factors=[0.6, 0.8, 1.0, 1.2, 1.5],
            low=0.0,
            high=5.0
        )

    # Min data in leaf
    if "model__min_data_in_leaf" in best_params:
        narrow_params["model__min_data_in_leaf"] = refine_int(
            best_params["model__min_data_in_leaf"],
            offsets=[2, 5, 10],
            low=1,
            high=50
        )

    # Random strength
    if "model__random_strength" in best_params:
        narrow_params["model__random_strength"] = refine_int(
            best_params["model__random_strength"],
            offsets=[1, 2, 4],
            low=0,
            high=20
        )

    # Freeze GPU-sensitive / discretization params
    if "model__border_count" in best_params:
        narrow_params["model__border_count"] = [
            best_params["model__border_count"]
        ]

    if "model__colsample_bylevel" in best_params:
        narrow_params["model__colsample_bylevel"] = [
            best_params["model__colsample_bylevel"]
        ]

    return narrow_params

def search_catboost_classification_v1(cfg):
    X_train, y_train = get_data(cfg)

    pipeline_1 = create_pipeline(cfg, "broad")

    logger.info("Starting broad hyperparameter search...")

    search_1 = search_best_params(
        pipeline_1,
        X_train,
        y_train,
        param_distributions_1,
        cfg,
        search_type="broad"
    )

    best_params_1 = search_1.best_params_

    param_distributions_2 = prepare_narrow_search_params(best_params_1)

    pipeline_2 = create_pipeline(cfg, "narrow")

    logger.info("Starting narrow hyperparameter search...")

    search_2 = search_best_params(
        pipeline_2,
        X_train,
        y_train,
        param_distributions_2,
        cfg,
        search_type="narrow"
    )

    best_params = search_2.best_params_
    
    return best_params
