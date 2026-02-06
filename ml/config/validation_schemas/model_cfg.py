from typing import Any, Dict, Optional

from ml.config.validation_schemas.model_specs import ModelSpecs
from ml.config.validation_schemas.search_cfg import SearchConfig
from ml.config.validation_schemas.train_cfg import TrainConfig

class SearchModelConfig(ModelSpecs):
    search: SearchConfig
    seed: int
    cv: int
    verbose: Optional[int] = 100
    training: Optional[Dict[str, Any]] = None  # inherited from shared defaults, unused

class TrainModelConfig(ModelSpecs):
    training: TrainConfig
    seed: int
    cv: int
    verbose: Optional[int] = 100
    search: Optional[Dict[str, Any]] = None  # inherited from shared defaults, unused