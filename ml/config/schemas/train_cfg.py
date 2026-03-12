"""Validation schema for model training-stage configuration."""

from pydantic import BaseModel, Field

from ml.config.schemas.base_model_params import BaseEnsembleParams, BaseModelParams
from ml.config.schemas.hardware_cfg import HardwareConfig, HardwareTaskType


class TrainConfig(BaseModel):
    """Training hyperparameters, hardware settings, and runtime controls."""

    iterations: int
    model: BaseModelParams = Field(default_factory=BaseModelParams)
    ensemble: BaseEnsembleParams = Field(default_factory=BaseEnsembleParams)
    hardware: HardwareConfig = Field(default_factory=lambda: HardwareConfig(task_type=HardwareTaskType.CPU))
    early_stopping_rounds: int = 0
    snapshot_interval_seconds: int = 600
