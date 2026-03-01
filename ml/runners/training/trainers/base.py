from abc import abstractmethod
from typing import Protocol

from ml.config.validation_schemas.model_cfg import TrainModelConfig
from ml.runners.training.constants.output import TRAIN_OUTPUT


class Trainer(Protocol):
    @abstractmethod
    def train(self, model_cfg: TrainModelConfig, strict: bool) -> TRAIN_OUTPUT: ...