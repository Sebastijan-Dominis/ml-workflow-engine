"""Schemas for the feature freezing process, including dependency versions and runtime information."""
from pydantic import BaseModel

from ml.modeling.models.runtime_info import Runtime


class FreezeDeps(BaseModel):
    """Schema for dependency versions used in the feature freezing process."""
    numpy: str
    pandas: str
    scikit_learn: str
    pyarrow: str
    pydantic: str
    PyYAML: str

class FreezeRuntimeInfo(BaseModel):
    """Schema for runtime information of the feature freezing process."""
    git_commit: str
    runtime_info: Runtime
    deps: FreezeDeps
    python_executable: str
