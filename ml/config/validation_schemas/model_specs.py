import logging
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pyparsing.helpers import Enum

from ml.exceptions import ConfigError
from ml.utils.experiments.class_weights.constants import \
    SUPPORTED_SCORING_FUNCTIONS

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
    include_in_model: bool = False
    filters: list[SegmentationFilter] = []

    # validate that if enabled is False, filters list must be empty, and include_in_model must be False, while if enabled is True, filters list must not be empty
    @field_validator("filters", mode="after")
    @classmethod
    def validate_filters_based_on_enabled(cls, v, info):
        enabled = info.data.get("enabled", False)
        if enabled and not v:
            msg = "Segmentation filters must be provided if segmentation is enabled."
            logger.error(msg)
            raise ConfigError(msg)
        if not enabled and v:
            msg = "Segmentation filters should not be provided if segmentation is disabled."
            logger.error(msg)
            raise ConfigError(msg)
        return v
    
    @field_validator("include_in_model", mode="after")
    @classmethod
    def validate_include_in_model_based_on_enabled(cls, v, info):
        enabled = info.data.get("enabled", False)
        if enabled and v is None:
            msg = "include_in_model must be specified if segmentation is enabled."
            logger.error(msg)
            raise ConfigError(msg)
        if not enabled and v:
            msg = "include_in_model should be False if segmentation is disabled."
            logger.error(msg)
            raise ConfigError(msg)
        return v

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

class ScoringConfig(BaseModel):
    policy: Literal["fixed", "adaptive_binary", "regression_default"] = "fixed"
    fixed_metric: Optional[SUPPORTED_SCORING_FUNCTIONS] = None
    pr_auc_threshold: Optional[float] = None

# Ensure that pr_auc_threshold is set if policy is adaptive_binary, and fixed_metric is set if policy is fixed
@field_validator("fixed_metric", mode="after")
def validate_fixed_metric_if_fixed_policy(cls, v, info):
    if info.data.get("policy") == "fixed" and v is None:
        msg = "fixed_metric must be specified if scoring policy is 'fixed'."
        logger.error(msg)
        raise ConfigError(msg)
    return v

@field_validator("pr_auc_threshold", mode="after")
def validate_pr_auc_threshold_if_adaptive_binary_policy(cls, v, info):
    if info.data.get("policy") == "adaptive_binary" and v is None:
        msg = "pr_auc_threshold must be specified if scoring policy is 'adaptive_binary'."
        logger.error(msg)
        raise ConfigError(msg)
    return v    

ClassImbalancePolicy = Literal[
    "off",          # never apply weighting
    "if_imbalanced",# apply if imbalance exceeds threshold
    "always"        # always apply
]

class ClassWeightingConfig(BaseModel):
    policy: ClassImbalancePolicy = "if_imbalanced"
    imbalance_threshold: float = 0.1  # e.g., if the minority class is less than 10% of the data, consider it imbalanced
    strategy: Literal["ratio", "balanced"] = "ratio"

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
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    split: SplitConfig
    algorithm: AlgorithmConfig
    model_class: str
    pipeline: PipelineConfig
    scoring: ScoringConfig
    class_weighting: ClassWeightingConfig = Field(default_factory=ClassWeightingConfig)
    feature_store: FeatureStoreConfig
    explainability: ExplainabilityConfig = Field(default_factory=ExplainabilityConfig)
    data_type: DATA_TYPE
    meta: MetaConfig = Field(default_factory=MetaConfig, alias="_meta")

    model_config = ConfigDict(extra="forbid", populate_by_name=True)
