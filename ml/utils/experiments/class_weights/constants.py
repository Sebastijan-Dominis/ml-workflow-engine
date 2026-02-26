from typing import Literal

SUPPORTED_SCORING_FUNCTIONS = Literal["roc_auc", "pr_auc", "rmse"]
SUPPORTED_LIBRARIES = Literal["xgboost", "lightgbm", "catboost", "sklearn"]