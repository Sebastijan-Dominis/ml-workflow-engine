import logging
logger = logging.getLogger(__name__)
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import RandomizedSearchCV

def get_data(cfg):
    try:
        features_path = cfg["features"]["features_path"]

        X_train = pd.read_parquet(Path(features_path) / "X_train.parquet")

        y_train = pd.read_parquet(Path(features_path) / "y_train.parquet")

        return X_train, y_train
    
    except Exception:
        logger.exception("Failed to load features data.")
        raise

def search_best_params(pipeline, X_train, y_train, param_distributions, cfg, search_type, error_score=np.nan):
    n_iter = cfg["search"][search_type]["n_iter"]
    cv = cfg["search"]["cv"]
    scoring = cfg["search"]["scoring"]
    random_state = cfg["search"]["random_state"]

    try:
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            verbose=2,
            n_jobs=1, # Use 1 to avoid potential GPU memory issues
            random_state=random_state,
            error_score=error_score
        )
    except Exception:
        logger.exception("Failed to initialize RandomizedSearchCV.")
        raise

    try:
        search.fit(X_train, y_train)
    except Exception:
        logger.exception("Failed to fit RandomizedSearchCV.")
        raise

    return search