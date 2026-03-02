"""Registry mapping algorithm families to training runner implementations."""

from ml.runners.training.trainers.catboost.catboost import TrainCatboost

TRAINERS = {
    "catboost": TrainCatboost
}

