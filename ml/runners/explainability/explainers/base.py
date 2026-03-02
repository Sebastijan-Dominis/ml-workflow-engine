"""Protocol definitions for explainability runner implementations."""

from abc import abstractmethod
from pathlib import Path
from typing import Protocol

from ml.config.validation_schemas.model_cfg import TrainModelConfig
from ml.runners.explainability.constants.output import ExplainabilityOutput

class Explainer(Protocol):
    """Structural interface implemented by explainability engines."""

    @abstractmethod
    def explain(self, *, model_cfg: TrainModelConfig, train_dir: Path, top_k: int) -> ExplainabilityOutput:
        """Generate explainability artifacts for a trained model run."""
        ...