"""Protocol definition for evaluation runner implementations."""

from abc import abstractmethod
from pathlib import Path
from typing import Optional, Protocol

from ml.config.schemas.model_cfg import TrainModelConfig
from ml.runners.evaluation.constants.output import EvaluateOutput


class Evaluator(Protocol):
    """Structural interface for task-specific evaluator implementations."""

    @abstractmethod
    def evaluate(self, *, model_cfg: TrainModelConfig, strict: bool, best_threshold: Optional[float], train_dir: Path) -> EvaluateOutput:
        """Evaluate a trained model and return standardized evaluation output."""
        ...