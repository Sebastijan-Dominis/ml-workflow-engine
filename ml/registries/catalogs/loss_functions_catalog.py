"""Mapping from sklearn regression scoring keys to model loss labels."""

REGRESSION_LOSS_FUNCTIONS = {
    "neg_root_mean_squared_error": "RMSE",
    "neg_mean_absolute_error": "MAE",
    "neg_mean_poisson_deviance": "Poisson"
}