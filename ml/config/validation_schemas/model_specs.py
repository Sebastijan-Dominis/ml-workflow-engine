import logging
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pyparsing.helpers import Enum

from ml.exceptions import ConfigError

logger = logging.getLogger(__name__)

class MetaConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    sources: Optional[Dict[str, Any]] = None
    env: Optional[str] = None
    best_params_path: Optional[str] = None
    validation_status: Optional[str] = None
    validation_errors: Optional[List[Any]] = None
    config_hash: Optional[str] = None

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

class FeatureSetConfig(BaseModel):
    ref: str
    name: str
    version: str
    schema_format: str
    input_schema: str
    derived_schema: str
    data_format: str
    X_train: str
    X_val: str
    X_test: str
    y_train: str
    y_val: str
    y_test: str

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

class ModelSpecs(BaseModel):
    problem: str
    segment: SegmentConfig
    version: str
    task: TaskConfig
    target: str
    algorithm: AlgorithmConfig
    model_class: str
    pipeline: PipelineConfig
    feature_store: FeatureStoreConfig
    explainability: ExplainabilityConfig = Field(default_factory=ExplainabilityConfig)
    data_type: Optional[str] = None
    meta: MetaConfig = Field(default_factory=MetaConfig, alias="_meta")

    model_config = ConfigDict(extra="forbid", populate_by_name=True)
