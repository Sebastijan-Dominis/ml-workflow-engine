from typing import Literal

from pydantic import BaseModel


class FeatureLineage(BaseModel):
    name: str
    version: str
    snapshot_id: str
    file_hash: str
    in_memory_hash: str
    feature_schema_hash: str
    operator_hash: str
    feature_type: Literal["tabular", "time-series"]
