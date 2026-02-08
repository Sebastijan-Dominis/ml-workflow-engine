from ml.runners.training.custom_training_scripts.train_catboost import train_catboost

TRAIN_REGISTRY = {
    "catboost": train_catboost
}