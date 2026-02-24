from abc import abstractmethod
from typing import Any, Protocol

from sklearn.pipeline import Pipeline

from ml.config.validation_schemas.model_cfg import TrainModelConfig
from ml.runners.training.constants.output import TRAIN_OUTPUT


class Trainer(Protocol):
    @abstractmethod
    def train(self, model_cfg: TrainModelConfig, strict: bool) -> TRAIN_OUTPUT: ...