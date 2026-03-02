"""Hardware execution configuration schemas for model workflows."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, field_validator, validator


# === Hardware TaskType enum ===
class HardwareTaskType(str, Enum):
    """Supported compute backends for model execution."""

    CPU = "CPU"
    GPU = "GPU"

# Normalize task_type to uppercase
class HardwareConfig(BaseModel):
    """Runtime hardware settings for search/training operations."""

    task_type: HardwareTaskType = HardwareTaskType.GPU if "SearchConfig" in __name__ else HardwareTaskType.CPU
    devices: list[int] = [0] if "SearchConfig" in __name__ else []
    memory_limit_gb: Optional[float] = None
    allow_growth: Optional[bool] = False

    @field_validator("task_type", mode="before")
    def normalize_task_type(cls, v):
        """Normalize task type values to uppercase prior to enum parsing.

        Args:
            cls: Pydantic model class invoking the validator.
            v: Raw ``task_type`` value supplied by configuration input.

        Returns:
            Uppercased string when ``v`` is text, otherwise the original value.
        """
        if isinstance(v, str):
            return v.upper()
        return v