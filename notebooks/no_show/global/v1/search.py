import logging
from pathlib import Path

import pandas as pd

from notebooks.common_functions import (
    get_best_f1_thresh,
    get_data,
    prepare_narrow_search_params,
    save_training_config,
    search_best_params,
)
from notebooks.logging_config import setup_logging
from notebooks.pipelines import create_classification_pipeline_1

logger = logging.getLogger(__name__)

setup_logging()

name = "no_show_global"
version = "v1"

try:
    feature_path = Path("data/features/no_show/global/v1")
    X_train, X_test, y_train, y_test = get_data(feature_path)
except Exception:
    logger.exception("Failed to load data.")
    raise

try:
    schema = pd.read_csv(Path("data/features/no_show/global/v1/schema.csv"))
except Exception:
    logger.exception("Failed to load schema.")
    raise

try:
    pipeline = create_classification_pipeline_1(schema, 800, ["total_stay", "adr_per_person", "arrival_season"])
except Exception:
    logger.exception("Failed to create pipeline.")
    raise

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

logger.info(f"Starting broad hyperparameter search for model {name}_{version}...")

try:
    search_1 = search_best_params(pipeline, X_train, y_train, param_distributions_1, False, "broad")
    best_params_1 = search_1.best_params_
except Exception:
    logger.exception("Failed during broad hyperparameter search.")
    raise

logger.info(f"Starting narrow hyperparameter search for model {name}_{version}...")

try:
    param_distributions_2 = prepare_narrow_search_params(best_params_1)
    search_2 = search_best_params(pipeline, X_train, y_train, param_distributions_2, False, "narrow")
    optimal_pipeline = search_2.best_estimator_
except Exception:
    logger.exception("Failed during narrow hyperparameter search.")
    raise

try:
    best_threshold = get_best_f1_thresh(optimal_pipeline, X_test, y_test)
except Exception:
    logger.exception("Failed to determine best F1 threshold.")
    raise

task = "binary_classification"
feature_path = "data/features/no_show/global/v1"
features_version = "v1"
target = "no_show"
algorithm = "catboost"
clean_params = {k.replace("model__", ""): v for k, v in search_2.best_params_.items()}
iterations = 2500
threshold = float(best_threshold)
pipeline_steps = {
    "validate_schema": True,
    "fill_categorical_missing": True,
    "feature_engineering": True,
    "feature_selection": True,
}

logger.info(f"Saving training configuration for model {name}_{version}...")

try:
    save_training_config(
        name, task, version, feature_path, features_version, target, algorithm,
        clean_params, iterations, threshold, pipeline_steps
    )
except Exception:
    logger.exception("Failed to save training configuration.")
    raise

logger.info(f"Hyperparameter search and configuration saving for model {name}_{version} completed successfully.")