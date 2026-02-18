from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    path: Path
    metadata_path: Path
    source: str
    format: Literal["csv","parquet","json", "arrow"]

class SegmentationFilter(BaseModel):
    column: str
    op: Literal["eq","neq","in","not_in","gt","gte","lt","lte"]
    value: int | str | list[int] | list[str]

class SegmentationConfig(BaseModel):
    enabled: bool = False
    filters: list[SegmentationFilter] = []

class ClassesConfig(BaseModel):
    count: int
    positive_class: int | str

class TargetConstraintsConfig(BaseModel):
    min_value: Optional[float] = None
    max_value: Optional[float] = None

class TargetConfig(BaseModel):
    name: str
    allowed_dtypes: list[str]
    problem_type: Literal["classification", "regression", "clustering"]
    classes: ClassesConfig
    constraints: TargetConstraintsConfig = Field(default_factory=TargetConstraintsConfig)

class ExcludeColumnsConfig(BaseModel):
    leaky: Optional[list[str]] = None
    useless: Optional[list[str]] = None

class ColumnConfig(BaseModel):
    include: list[str]
    exclude: ExcludeColumnsConfig = Field(default_factory=ExcludeColumnsConfig)

class FeatureRolesConfig(BaseModel):
    categorical: list[str]
    numerical: list[str]
    datetime: list[str]

class OperatorsConfig(BaseModel):
    mode: Literal["materialized","logical"]
    list: list[str]
    hash: str

class SplitConfig(BaseModel):
    strategy: Literal["random"]
    stratify_by: str
    test_size: float = Field(gt=0.0, lt=1.0)
    val_size: float = Field(gt=0.0, lt=1.0)
    random_state: int

class ConstraintsConfig(BaseModel):
    forbid_nulls: list[str]
    max_cardinality: dict[str, int]

class StorageConfig(BaseModel):
    format: Literal["parquet"]
    compression: Optional[str] = "snappy"

class LineageConfig(BaseModel):
    source_datasets: list[str]
    feature_set_version: str
    created_by: str
    created_at: str

class TabularFeaturesConfig(BaseModel):
    type: str = "tabular"
    description: str | None = None
    data: DataConfig
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    min_rows: int = Field(default=1000, ge=0)
    min_class_count: int = Field(default=10, ge=0)
    feature_store_path: Path
    target: TargetConfig
    columns: ColumnConfig
    feature_roles: FeatureRolesConfig
    operators: Optional[OperatorsConfig] = None
    split: SplitConfig
    constraints: ConstraintsConfig
    storage: StorageConfig
    lineage: LineageConfig
    ...
    class Config:
        extra = "forbid"