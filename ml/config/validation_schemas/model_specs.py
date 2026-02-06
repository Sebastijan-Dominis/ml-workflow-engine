import logging
logger = logging.getLogger(__name__)
from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import Optional, Dict, Any, List
from pyparsing import Enum


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


class ExplainabilityMethodConfig(BaseModel):
    enabled: bool = True
    params: Dict[str, Any] = Field(default_factory=dict)  # e.g., {"type": "PredictionValuesChange"}

class ExplainabilityConfig(BaseModel):
    enabled: bool = True
    top_k: int = 20
    methods: Dict[str, ExplainabilityMethodConfig] = Field(default_factory=lambda: {
        "feature_importances": ExplainabilityMethodConfig(enabled=True, params={"type": "PredictionValuesChange"}),
        "shap": ExplainabilityMethodConfig(enabled=True, params={"approximate": "tree"})
    })

    @field_validator("methods", mode="before")
    def validate_methods(cls, v):
        allowed = {"feature_importances", "shap"}
        invalid = set(v.keys()) - allowed
        if invalid:
            msg = f"Unsupported explainability method(s): {', '.join(invalid)}"
            logger.error(msg)
            raise ValueError(msg)
        return v

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
