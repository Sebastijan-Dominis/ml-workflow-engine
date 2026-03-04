"""Registry listing supported hyperparameter names per algorithm family."""

MODEL_PARAM_REGISTRY = {
    "catboost": ["depth", "learning_rate", "l2_leaf_reg", "random_strength", "min_data_in_leaf", "bagging_temperature", "border_count", "colsample_bylevel"],
}
