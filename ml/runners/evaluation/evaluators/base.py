from pathlib import Path

from abc import abstractmethod
from typing import Protocol, Optional

from ml.config.validation_schemas.model_cfg import TrainModelConfig


class Evaluator(Protocol):
    @abstractmethod
    def evaluate(self, *, model_cfg: TrainModelConfig, strict: bool, best_threshold: Optional[float], train_dir: Path) -> tuple[dict[str, dict[str, float]], list[dict]]: ...