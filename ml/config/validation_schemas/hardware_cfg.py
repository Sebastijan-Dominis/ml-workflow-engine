from enum import Enum
from pydantic import BaseModel, field_validator, validator
from typing import List, Optional

# === Hardware TaskType enum ===
class HardwareTaskType(str, Enum):
    CPU = "CPU"
    GPU = "GPU"

# Normalize task_type to uppercase
class HardwareConfig(BaseModel):
    task_type: HardwareTaskType = HardwareTaskType.GPU if "SearchConfig" in __name__ else HardwareTaskType.CPU
    devices: List[int] = [0] if "SearchConfig" in __name__ else []
    memory_limit_gb: Optional[float] = None
    allow_growth: Optional[bool] = False

    @field_validator("task_type", mode="before")
    def normalize_task_type(cls, v):
        if isinstance(v, str):
            return v.upper()
        return v