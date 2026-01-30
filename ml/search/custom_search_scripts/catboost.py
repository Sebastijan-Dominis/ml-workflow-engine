import logging
logger = logging.getLogger(__name__)
from catboost import CatBoostClassifier, CatBoostRegressor
import yaml
from pathlib import Path

from ml.search.utils import load_train_data, search_best_params
from ml.utils import get_cat_features, load_schemas
from ml.pipelines.builders import build_pipeline
from ml.registry import MODEL_REGISTRY

param_distributions_1 = {
    'Model__depth': [4, 6, 8, 10],
    'Model__learning_rate': [0.005, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3],
    'Model__l2_leaf_reg': [1, 3, 5, 7, 9],
    'Model__bagging_temperature': [0, 0.25, 0.5, 0.75, 1, 2, 5],
    'Model__border_count': [32, 64, 128, 254],
    'Model__min_data_in_leaf': [1, 3, 5, 8, 10, 15, 20],
    # 'Model__colsample_bylevel': [0.6, 0.8, 1.0], # Ok to explore when using CPU
    'Model__colsample_bylevel': [1.0], # Set to 1.0 to avoid issues with GPU training
    'Model__random_strength': [1, 2, 5, 8, 10, 12]
}

def prepare_model(cfg_model_specs, cfg_search, search_phase, cat_features):
    model = MODEL_REGISTRY[cfg_model_specs["model_class"]](
        # Basic hyperparameters
        iterations=cfg_search[search_phase]["iterations"],
        task_type='GPU',             # enable GPU
        devices='0',                 # GPU device
        verbose=100,                
        random_state=cfg_model_specs['seed'],
        cat_features=cat_features,
        class_weights=cfg_model_specs.get("class_weights", None)
    )
    return model

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
    if "Model__depth" in best_params:
        narrow_params["Model__depth"] = refine_int(
            best_params["Model__depth"],
            offsets=[1, 2],
            low=2,
            high=12
        )

    # Learning rate
    if "Model__learning_rate" in best_params:
        narrow_params["Model__learning_rate"] = refine_float_mult(
            best_params["Model__learning_rate"],
            factors=[0.7, 0.85, 1.0, 1.15, 1.3],
            low=0.003,
            high=0.5
        )

    # L2 regularization
    if "Model__l2_leaf_reg" in best_params:
        narrow_params["Model__l2_leaf_reg"] = refine_int(
            best_params["Model__l2_leaf_reg"],
            offsets=[1, 2, 3],
            low=1,
            high=30
        )

    # Bagging temperature
    if "Model__bagging_temperature" in best_params:
        narrow_params["Model__bagging_temperature"] = refine_float_mult(
            best_params["Model__bagging_temperature"],
            factors=[0.6, 0.8, 1.0, 1.2, 1.5],
            low=0.0,
            high=5.0
        )

    # Min data in leaf
    if "Model__min_data_in_leaf" in best_params:
        narrow_params["Model__min_data_in_leaf"] = refine_int(
            best_params["Model__min_data_in_leaf"],
            offsets=[2, 5, 10],
            low=1,
            high=50
        )

    # Random strength
    if "Model__random_strength" in best_params:
        narrow_params["Model__random_strength"] = refine_int(
            best_params["Model__random_strength"],
            offsets=[1, 2, 4],
            low=0,
            high=20
        )

    # Freeze GPU-sensitive / discretization params
    if "Model__border_count" in best_params:
        narrow_params["Model__border_count"] = [
            best_params["Model__border_count"]
        ]

    if "Model__colsample_bylevel" in best_params:
        narrow_params["Model__colsample_bylevel"] = [
            best_params["Model__colsample_bylevel"]
        ]

    return narrow_params

def search_catboost(cfg_model_specs: dict, cfg_search: dict) -> dict:
    features_path = Path(cfg_model_specs["features"]["path"])
    X_train_file_name = cfg_model_specs["features"]["X_train"]
    y_train_file_name = cfg_model_specs["features"]["y_train"]

    raw_schema, derived_schema = load_schemas(features_path, logger)
    X_train, y_train = load_train_data(features_path, X_train_file_name, y_train_file_name)

    pipeline_1 = build_pipeline(
        pipeline_cfg=yaml.safe_load(
            open(f"configs/pipelines/{cfg_model_specs['pipeline']}.yaml")
        ),
        raw_schema=raw_schema,
        derived_schema=derived_schema,
    )

    cat_features = get_cat_features(raw_schema, derived_schema)

    model_1 = prepare_model(cfg_model_specs, cfg_search, "broad", cat_features)

    if not isinstance(model_1, CatBoostClassifier or CatBoostRegressor):
        logger.error("Defined model is not a CatBoostClassifier or CatBoostRegressor instance.")
        raise TypeError("Defined model is not a CatBoostClassifier or CatBoostRegressor instance.")

    pipeline_1.steps.append(("Model", model_1))

    logger.info("Starting broad hyperparameter search...")

    search_1 = search_best_params(
        pipeline_1,
        X_train,
        y_train,
        param_distributions_1,
        cfg_search,
        search_type="broad"
    )

    best_params_1 = search_1.best_params_

    param_distributions_2 = prepare_narrow_search_params(best_params_1)

    pipeline_2 = build_pipeline(
        pipeline_cfg=yaml.safe_load(
            open(f"configs/pipelines/{cfg_model_specs['pipeline']}.yaml")
        ),
        raw_schema=raw_schema,
        derived_schema=derived_schema,
    )

    model_2 = prepare_model(cfg_model_specs, cfg_search, "narrow", cat_features)

    pipeline_2.steps.append(("Model", model_2))

    logger.info("Starting narrow hyperparameter search...")

    search_2 = search_best_params(
        pipeline_2,
        X_train,
        y_train,
        param_distributions_2,
        cfg_search,
        search_type="narrow"
    )

    best_params = search_2.best_params_
    
    return best_params
