import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import is_classifier
from sklearn.model_selection import RandomizedSearchCV, check_cv
from sklearn.pipeline import Pipeline

from ml.config.validation_schemas.model_cfg import SearchModelConfig
from ml.search.constants import SEARCH_PHASES
from ml.utils.experiments.class_weights.constants import \
    SUPPORTED_SCORING_FUNCTIONS

logger = logging.getLogger(__name__)

def perform_randomized_search(
    pipeline: Pipeline, 
    *,
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    param_distributions: dict[str, Any], 
    model_cfg: SearchModelConfig, 
    search_phase: SEARCH_PHASES,
    scoring: SUPPORTED_SCORING_FUNCTIONS
) -> dict[str, Any]:
    search_phase_cfg = getattr(model_cfg.search, search_phase)
    n_iter = search_phase_cfg.n_iter
    cv = model_cfg.cv
    verbose = model_cfg.verbose if model_cfg.verbose is not None else 100
    hardware = model_cfg.search.hardware
    n_jobs = 1 if hardware and hardware.task_type.value == "GPU" else -1
    random_state = model_cfg.search.random_state
    error_score = model_cfg.search.error_score if model_cfg.search.error_score is not None else np.nan

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        verbose=verbose,
        n_jobs=n_jobs,
        random_state=random_state,
        error_score=error_score
    )

    # Ensuring accurate logging
    resolved_cv = check_cv(
        cv,
        y_train,
        classifier=is_classifier(pipeline)
    )

    cv_type = cv.__class__.__name__ if not isinstance(cv, int) and cv is not None else resolved_cv.__class__.__name__

    n_splits = resolved_cv.get_n_splits(X_train, y_train)

    logger.info(
        "Randomized search using param_distributions: %s, n_iter: %d, cv_type: %s, n_splits: %d, scoring: %s, verbose: %d, n_jobs: %d, random_state: %s, error_score: %s",
        param_distributions,
        n_iter,
        cv_type,
        n_splits,
        scoring,
        verbose,
        n_jobs,
        random_state,
        error_score
    )

    logger.info(f"Performing {search_phase} hyperparameter search...")

    search.fit(X_train, y_train)

    return {
        "best_params": search.best_params_,
        "best_score": search.best_score_,
        "best_index": int(search.best_index_),
        "cv_results": {
            "mean_test_score": search.cv_results_["mean_test_score"].tolist(),
            "std_test_score": search.cv_results_["std_test_score"].tolist(),
            "rank_test_score": search.cv_results_["rank_test_score"].tolist(),
        },
        "param_distributions": param_distributions,
        "n_iter": n_iter,
        "cv": cv if isinstance(cv, int) else cv.__class__.__name__,
        "scoring": scoring,
        "random_state": random_state,
        "error_score": str(error_score),
        "search_phase": search_phase,
    }

