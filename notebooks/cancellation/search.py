from sklearn.model_selection import RandomizedSearchCV
import numpy as np

def search_best_params(pipeline, X_train, y_train, param_distributions, search_type, cv=3, scoring="roc_auc", error_score=np.nan):
    
    if search_type == "broad":
        n_iter = 5
    elif search_type == "narrow":
        n_iter = 8
    else:
        raise ValueError("search_type must be either 'broad' or 'narrow'")
    
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=1,
        cv=2,
        scoring=scoring,
        verbose=2,
        n_jobs=1, # Use 1 to avoid potential GPU memory issues
        random_state=42,
        error_score=error_score
    )

    search.fit(X_train, y_train)

    return search