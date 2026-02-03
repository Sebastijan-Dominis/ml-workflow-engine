import logging
logger = logging.getLogger(__name__)
import numpy as np
from sklearn.model_selection import RandomizedSearchCV

def perform_randomized_search(pipeline, X_train, y_train, param_distributions, model_cfg, search_type):
    n_iter = model_cfg["search"][search_type]["n_iter"]
    cv = model_cfg["cv"]
    scoring = model_cfg["search"]["scoring"]
    verbose = model_cfg.get("verbose", 100)
    n_jobs = 1 if model_cfg.get("search", {}).get("hardware", {}).get("task_type", "CPU") == "GPU" else -1
    random_state = model_cfg["search"]["random_state"]
    error_score = model_cfg["search"].get("error_score", np.nan)

    logger.info("Using CV type: %s", cv if isinstance(cv, int) else cv.__class__.__name__)
    logger.info("n_jobs set to: %d", n_jobs)

    try:
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
    except Exception:
        logger.exception("Failed to initialize RandomizedSearchCV.")
        raise

    try:
        search.fit(X_train, y_train)
    except Exception:
        logger.exception("Failed to fit RandomizedSearchCV.")
        raise

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
        "search_type": search_type,
    }

