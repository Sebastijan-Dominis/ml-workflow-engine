"""Type constraints for supported scoring metrics and ML libraries."""

from typing import Literal

# Average precision is basically the same as pr_auc, but it uses a different method to calculate the area under the curve. It calculates the area using a step function. The terms are used interchangeably in this project.
SUPPORTED_SCORING_FUNCTIONS = Literal["roc_auc", "average_precision", "neg_root_mean_squared_error", "neg_mean_absolute_error", "neg_mean_poisson_deviance"]
SUPPORTED_LIBRARIES = Literal["xgboost", "lightgbm", "catboost", "sklearn"]
