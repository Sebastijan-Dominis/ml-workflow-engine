import logging
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pyparsing.helpers import Enum

from ml.config.validation_schemas.data import DataConfig
from ml.exceptions import ConfigError

logger = logging.getLogger(__name__)

class SegmentConfig(BaseModel):
    name: str
    description: Optional[str] = None

class TaskType(str, Enum):
    classification = "classification"
    regression = "regression"
    ranking = "ranking"
    time_series = "time_series"

class TaskConfig(BaseModel):
    type: TaskType
    subtype: Optional[str] = None

    @field_validator("type", mode="before")
    def normalize_task_type(cls, v):
        if isinstance(v, str):
            return v.lower()
        return v

class ClassesConfig(BaseModel):
    count: int
    positive_class: int | str
    min_class_count: int

class TargetConstraintsConfig(BaseModel):
    min_value: Optional[float] = None
    max_value: Optional[float] = None

class TargetConfig(BaseModel):
    name: str
    allowed_dtypes: list[str]
    problem_type: Literal["classification", "regression"]
    classes: Optional[ClassesConfig] = None
    constraints: TargetConstraintsConfig = Field(default_factory=TargetConstraintsConfig)
    version: str

    @field_validator("classes", mode="after")
    @classmethod
    def validate_classes_for_classification(cls, v, info):
        if info.data.get("problem_type") == "classification" and v is None:
            msg = "Classes must be provided for classification problems."
            logger.error(msg)
            raise ConfigError(msg)
        if info.data.get("problem_type") == "classification" and v is not None and v.count < 2:
            msg = f"Classes count must be at least 2 for classification problems, got {v.count}."
            logger.error(msg)
            raise ConfigError(msg)
        return v
    
    @field_validator("version", mode="before")
    @classmethod
    def validate_version_format(cls, v):
        if not isinstance(v, str) or not v.startswith("v") or not v[1:].isdigit():
            msg = f"Version must be in format 'v{{number}}', e.g. 'v1', 'v2', etc. Got '{v}'."
            logger.error(msg)
            raise ConfigError(msg)
        return v

class SegmentationFilter(BaseModel):
    column: str
    op: Literal["eq","neq","in","not_in","gt","gte","lt","lte"]
    value: int | str | list[int] | list[str]

class SegmentationConfig(BaseModel):
    enabled: bool = False
    filters: list[SegmentationFilter] = []

class FeatureSetConfig(BaseModel):
    name: str
    version: str
    schema_format: str
    input_schema: str
    derived_schema: str
    data_format: str
    file_name: str

class SplitConfig(BaseModel):
    strategy: Literal["random"]
    stratify_by: str
    test_size: float = Field(gt=0.0, lt=1.0)
    val_size: float = Field(gt=0.0, lt=1.0)
    random_state: int

class AlgorithmConfig(str, Enum):
    catboost = "catboost"
    xgboost = "xgboost"
    lightgbm = "lightgbm"
    random_forest = "random_forest"
    logistic_regression = "logistic_regression"
    neural_network = "neural_network"
    prophet = "prophet"

class FeatureStoreConfig(BaseModel):
    path: str
    feature_sets: list[FeatureSetConfig]

class PipelineConfig(BaseModel):
    version: str
    path: str

class FeatureImportanceMethodConfig(BaseModel):
    enabled: bool = False
    type: Optional[Literal["PredictionValuesChange", "LossFunctionChange", "FeatureImportance", "TotalGain"]] = None

@field_validator("type", mode="after")
def validate_type_if_enabled(cls, v, info):
    if info.data.get("enabled") and v is None:
        msg = "Type must be specified if feature importance method is enabled."
        logger.error(msg)
        raise ConfigError(msg)
    return v

class SHAPMethodConfig(BaseModel):
    enabled: bool = False
    approximate: Optional[Literal["tree", "linear", "kernel"]] = None

@field_validator("approximate", mode="after")
def validate_approximate_if_enabled(cls, v, info):
    if info.data.get("enabled") and v is None:
        msg = "Approximate method must be specified if SHAP method is enabled."
        logger.error(msg)
        raise ConfigError(msg)
    return v

class ExplainabilityMethodsConfig(BaseModel):
    feature_importances: FeatureImportanceMethodConfig = Field(default_factory=FeatureImportanceMethodConfig)
    shap: SHAPMethodConfig = Field(default_factory=SHAPMethodConfig)

class ExplainabilityConfig(BaseModel):
    enabled: bool = True
    top_k: int = 20
    methods: ExplainabilityMethodsConfig = Field(default_factory=ExplainabilityMethodsConfig)

DATA_TYPE = Literal["tabular", "time-series"]

class MetaConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    sources: Optional[Dict[str, Any]] = None
    env: Optional[str] = None
    best_params_path: Optional[str] = None
    validation_status: Optional[str] = None
    validation_errors: Optional[List[Any]] = None
    config_hash: Optional[str] = None

class ModelSpecs(BaseModel):
    problem: str
    segment: SegmentConfig
    version: str
    task: TaskConfig
    target: TargetConfig
    data: DataConfig
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    split: SplitConfig
    algorithm: AlgorithmConfig
    model_class: str
    pipeline: PipelineConfig
    feature_store: FeatureStoreConfig
    explainability: ExplainabilityConfig = Field(default_factory=ExplainabilityConfig)
    data_type: DATA_TYPE
    meta: MetaConfig = Field(default_factory=MetaConfig, alias="_meta")

    model_config = ConfigDict(extra="forbid", populate_by_name=True)
