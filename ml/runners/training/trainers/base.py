from abc import abstractmethod
from typing import Any, Protocol

from sklearn.pipeline import Pipeline

from ml.config.validation_schemas.model_cfg import TrainModelConfig


class Trainer(Protocol):
    @abstractmethod
    def train(self, model_cfg: TrainModelConfig, strict: bool) -> tuple[Any, Pipeline, list[dict], dict[str, float], str | None]: ...