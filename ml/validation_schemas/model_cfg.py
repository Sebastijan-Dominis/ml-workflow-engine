from typing import Optional

from ml.validation_schemas.model_specs import ModelSpecs
from ml.validation_schemas.search_cfg import SearchConfig
from ml.validation_schemas.train_cfg import TrainConfig

class SearchModelConfig(ModelSpecs):
    search: SearchConfig
    seed: int
    cv: int
    verbose: Optional[int] = 100

class TrainModelConfig(ModelSpecs):
    training: TrainConfig
    seed: int
    cv: int
    verbose: Optional[int] = 100