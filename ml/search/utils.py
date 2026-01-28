import logging
logger = logging.getLogger(__name__)
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import RandomizedSearchCV

def get_data(features_path: Path) -> tuple:
    try:
        X_train = pd.read_parquet(Path(features_path) / "X_train.parquet")

        y_train = pd.read_parquet(Path(features_path) / "y_train.parquet")

        return X_train, y_train
    
    except Exception:
        logger.exception("Failed to load features data.")
        raise

def search_best_params(pipeline, X_train, y_train, param_distributions, cfg_search, search_type, error_score=np.nan):
    n_iter = cfg_search[search_type]["n_iter"]
    cv = cfg_search["cv"]
    scoring = cfg_search["scoring"]
    random_state = cfg_search["random_state"]
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

def load_schemas(features_path: Path) -> tuple:
    raw_schema_path = features_path / "schema.csv"
    derived_schema_path = features_path / "derived_schema.csv"
    try:
        raw_schema = pd.read_csv(raw_schema_path)
        derived_schema = pd.read_csv(derived_schema_path)
        return raw_schema, derived_schema
    except Exception:
        logger.exception(f"Failed to load schemas from {features_path}.")
        raise


def get_cat_features(raw_schema: pd.DataFrame, derived_schema: pd.DataFrame) -> list:
    raw_categoricals = raw_schema.loc[
        raw_schema["dtype"].isin(["object", "string", "category"]),
        "feature",
    ].tolist()

    derived_categoricals = derived_schema.loc[
        derived_schema["dtype"].isin(["object", "string", "category"]),
        "feature",
    ].tolist()

    return raw_categoricals + derived_categoricals