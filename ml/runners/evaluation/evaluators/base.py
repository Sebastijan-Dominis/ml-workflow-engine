from abc import abstractmethod
from pathlib import Path
from typing import Optional, Protocol

import pandas as pd

from ml.config.validation_schemas.model_cfg import TrainModelConfig

class Evaluator(Protocol):
    @abstractmethod
    def evaluate(self, *, model_cfg: TrainModelConfig, strict: bool, best_threshold: Optional[float], train_dir: Path) -> tuple[dict[str, dict[str, float]], dict[str, pd.DataFrame], list[dict]]: ...