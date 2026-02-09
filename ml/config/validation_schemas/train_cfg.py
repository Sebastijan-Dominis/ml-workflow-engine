from typing import Any

from pydantic import BaseModel, Field

from ml.config.validation_schemas.base_model_params import BaseEnsembleParams, BaseModelParams
from ml.config.validation_schemas.hardware_cfg import HardwareConfig


class TrainConfig(BaseModel):
    iterations: int
    model: BaseModelParams = Field(default_factory=BaseModelParams)
    ensemble: BaseEnsembleParams = Field(default_factory=BaseEnsembleParams)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    early_stopping_rounds: int = 0