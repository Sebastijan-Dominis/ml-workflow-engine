"""Utilities for running randomized hyperparameter searches."""

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import is_classifier
from sklearn.model_selection import RandomizedSearchCV, check_cv
from sklearn.pipeline import Pipeline

from ml.config.schemas.model_cfg import SearchModelConfig
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
    """Run randomized search and return serializable best-result metadata.

    Args:
        pipeline: Estimator pipeline used for randomized search.
        X_train: Training feature matrix.
        y_train: Training target vector.
        param_distributions: Hyperparameter distributions for randomized sampling.
        model_cfg: Validated search model configuration.
        search_phase: Search phase identifier (broad or narrow).
        scoring: Supported scoring method used by the search.

    Returns:
        Serializable dictionary containing best parameters, scores, and CV summary.

    Raises:
        ValueError: Propagated when search inputs are invalid (for example,
            incompatible parameter distributions or CV configuration).

    Notes:
        For GPU execution, ``n_jobs`` is forced to ``1`` to avoid parallel worker
        contention with GPU-bound estimators.

    Side Effects:
        Fits ``pipeline`` via ``RandomizedSearchCV.fit`` and emits detailed run
        configuration logs.

    Examples:
        >>> results = perform_randomized_search(
        ...     pipeline,
        ...     X_train=X_train,
        ...     y_train=y_train,
        ...     param_distributions=param_distributions,
        ...     model_cfg=model_cfg,
        ...     search_phase="broad",
        ...     scoring="roc_auc",
        ... )
    """

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

