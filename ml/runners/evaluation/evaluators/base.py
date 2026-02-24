from abc import abstractmethod
from pathlib import Path
from typing import Optional, Protocol

from ml.config.validation_schemas.model_cfg import TrainModelConfig
from ml.runners.evaluation.constants.output import EVALUATE_OUTPUT


class Evaluator(Protocol):
    @abstractmethod
    def evaluate(self, *, model_cfg: TrainModelConfig, strict: bool, best_threshold: Optional[float], train_dir: Path) -> EVALUATE_OUTPUT: ...