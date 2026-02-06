from typing import Optional
from pydantic import BaseModel, Field

from ml.config.validation_schemas.hardware_cfg import HardwareConfig
from ml.config.validation_schemas.base_model_params import BaseModelParams, BaseEnsembleParams

class TrainConfig(BaseModel):
    iterations: int
    model: BaseModelParams = Field(default_factory=BaseModelParams)
    ensemble: BaseEnsembleParams = Field(default_factory=BaseEnsembleParams)
    hardware: Optional[HardwareConfig] = None