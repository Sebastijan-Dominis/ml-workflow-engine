"""Canonical Pydantic schemas for model specification configuration.

This module defines shared model-spec structures and validation rules used by
both search and training configuration entrypoints.
"""

import logging
from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import (BaseModel, ConfigDict, Field, field_validator,
                      model_validator)

from ml.exceptions import ConfigError
from ml.utils.experiments.class_weights.constants import \
    SUPPORTED_SCORING_FUNCTIONS

logger = logging.getLogger(__name__)

class SegmentConfig(BaseModel):
    """Model segment identifier and optional description."""

    name: str
    description: Optional[str] = None

class TaskType(str, Enum):
    """Supported high-level ML task categories."""

    classification = "classification"
    regression = "regression"
    ranking = "ranking"
    time_series = "time_series"

class TaskConfig(BaseModel):
    """Task type metadata for model execution and validation logic."""

    type: TaskType
    subtype: Optional[str] = None

    @field_validator("type", mode="before")
    def normalize_task_type(cls, v):
        """Normalize task type strings to lowercase before enum coercion.

        Args:
            v: Raw task type value.

        Returns:
            Any: Normalized task type value.
        """

        if isinstance(v, str):
            return v.lower()
        return v

class ClassesConfig(BaseModel):
    """Class metadata for classification targets."""

    count: int
    positive_class: int | str
    min_class_count: int

class TargetConstraintsConfig(BaseModel):
    """Numeric constraints applicable to target values."""

    min_value: Optional[float] = None
    max_value: Optional[float] = None

class TargetTransformConfig(BaseModel):
    """Optional target transformation settings."""

    enabled: bool = False
    type: Literal["log1p", "sqrt"] | None = None
    lambda_value: Optional[float] = None  # Only used for Box-Cox transform

    # Validate that if type is boxcox, then lambda_value must be provided, and if type is not boxcox, then lambda_value must be None
    @field_validator("lambda_value", mode="after")
    def validate_lambda_value_based_on_type(cls, v, info):
        """Validate transformation-specific lambda usage constraints.

        Args:
            v: Lambda value candidate.
            info: Pydantic field-validation context.

        Returns:
            Any: Validated lambda value.
        """

        transform_type = info.data.get("type")
        if transform_type == "boxcox" and v is None:
            msg = "lambda_value must be provided for Box-Cox transformation."
            logger.error(msg)
            raise ConfigError(msg)
        if transform_type != "boxcox" and v is not None:
            msg = "lambda_value should only be provided for Box-Cox transformation."
            logger.error(msg)
            raise ConfigError(msg)
        return v

class TargetConfig(BaseModel):
    """Target definition, datatype constraints, and transform options."""

    name: str
    version: str
    allowed_dtypes: list[str]
    classes: Optional[ClassesConfig] = None
    constraints: TargetConstraintsConfig = Field(default_factory=TargetConstraintsConfig)
    transform: TargetTransformConfig = Field(default_factory=TargetTransformConfig)

    @field_validator("version", mode="before")
    @classmethod
    def validate_version_format(cls, v):
        """Ensure target version follows ``v{number}`` convention.

        Args:
            v: Raw target version value.

        Returns:
            str: Validated target version string.
        """

        if not isinstance(v, str) or not v.startswith("v") or not v[1:].isdigit():
            msg = f"Version must be in format 'v{{number}}', e.g. 'v1', 'v2', etc. Got '{v}'."
            logger.error(msg)
            raise ConfigError(msg)
        return v

class SegmentationFilter(BaseModel):
    """Single logical filter rule for dataset segmentation."""

    column: str
    op: Literal["eq","neq","in","not_in","gt","gte","lt","lte"]
    value: int | str | list[int] | list[str]

class SegmentationConfig(BaseModel):
    """Segmentation enablement and filter configuration."""

    enabled: bool = False
    include_in_model: bool = False
    filters: list[SegmentationFilter] = []

    # validate that if enabled is False, filters list must be empty, and include_in_model must be False, while if enabled is True, filters list must not be empty
    @field_validator("filters", mode="after")
    @classmethod
    def validate_filters_based_on_enabled(cls, v, info):
        """Validate filter presence based on segmentation enablement.

        Args:
            v: Segmentation filters list.
            info: Pydantic field-validation context.

        Returns:
            list: Validated filters list.
        """

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
        """Validate model-include flag consistency with segmentation state.

        Args:
            v: Include-in-model flag value.
            info: Pydantic field-validation context.

        Returns:
            bool: Validated include-in-model flag.
        """

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
    """Feature-set artifact references in the feature store."""

    name: str
    version: str
    schema_format: str
    input_schema: str
    derived_schema: str
    data_format: str
    file_name: str

class SplitConfig(BaseModel):
    """Train/validation/test split strategy settings."""

    strategy: Literal["random"]
    stratify_by: Optional[str] = None
    test_size: float = Field(gt=0.0, lt=1.0)
    val_size: float = Field(gt=0.0, lt=1.0)
    random_state: int

class AlgorithmConfig(str, Enum):
    """Supported algorithm families."""

    catboost = "catboost"
    xgboost = "xgboost"
    lightgbm = "lightgbm"
    random_forest = "random_forest"
    logistic_regression = "logistic_regression"
    neural_network = "neural_network"
    prophet = "prophet"

class FeatureStoreConfig(BaseModel):
    """Feature store location and required feature set references."""

    path: str
    feature_sets: list[FeatureSetConfig]

class PipelineConfig(BaseModel):
    """Pipeline artifact metadata."""

    version: str
    path: str

class ScoringConfig(BaseModel):
    """Metric-scoring policy and optional thresholds."""

    policy: Literal["fixed", "adaptive_binary", "regression_default"] = "fixed"
    fixed_metric: Optional[SUPPORTED_SCORING_FUNCTIONS] = None
    pr_auc_threshold: Optional[float] = None

    # Ensure that pr_auc_threshold is set if policy is adaptive_binary, and fixed_metric is set if policy is fixed
    @field_validator("fixed_metric", mode="after")
    def validate_fixed_metric_if_fixed_policy(cls, v, info):
        """Ensure fixed scoring policy specifies a fixed metric.

        Args:
            v: Fixed metric value.
            info: Pydantic field-validation context.

        Returns:
            Any: Validated fixed metric.
        """

        if info.data.get("policy") == "fixed" and v is None:
            msg = "fixed_metric must be specified if scoring policy is 'fixed'."
            logger.error(msg)
            raise ConfigError(msg)
        return v

    @field_validator("pr_auc_threshold", mode="after")
    def validate_pr_auc_threshold_if_adaptive_binary_policy(cls, v, info):
        """Ensure adaptive-binary policy specifies PR-AUC threshold.

        Args:
            v: PR-AUC threshold value.
            info: Pydantic field-validation context.

        Returns:
            Any: Validated PR-AUC threshold.
        """

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
    """Class-weighting policy for classification imbalance handling."""

    policy: ClassImbalancePolicy = "off"
    imbalance_threshold: Optional[float] = None
    strategy: Optional[Literal["ratio", "balanced"]] = None

class FeatureImportanceMethodConfig(BaseModel):
    """Configuration for feature-importance explainability method."""

    enabled: bool = False
    type: Optional[Literal["PredictionValuesChange", "LossFunctionChange", "FeatureImportance", "TotalGain"]] = None

    @field_validator("type", mode="after")
    def validate_type_if_enabled(cls, v, info):
        """Require feature-importance method type when enabled.

        Args:
            v: Feature-importance type value.
            info: Pydantic field-validation context.

        Returns:
            Any: Validated feature-importance type.
        """

        if info.data.get("enabled") and v is None:
            msg = "Type must be specified if feature importance method is enabled."
            logger.error(msg)
            raise ConfigError(msg)
        return v

class SHAPMethodConfig(BaseModel):
    """Configuration for SHAP-based explainability method."""

    enabled: bool = False
    approximate: Optional[Literal["tree", "linear", "kernel"]] = None

    @field_validator("approximate", mode="after")
    def validate_approximate_if_enabled(cls, v, info):
        """Require SHAP approximation mode when SHAP is enabled.

        Args:
            v: SHAP approximation mode value.
            info: Pydantic field-validation context.

        Returns:
            Any: Validated SHAP approximation mode.
        """

        if info.data.get("enabled") and v is None:
            msg = "Approximate method must be specified if SHAP method is enabled."
            logger.error(msg)
            raise ConfigError(msg)
        return v

class ExplainabilityMethodsConfig(BaseModel):
    """Container for explainability method-level settings."""

    feature_importances: FeatureImportanceMethodConfig = Field(default_factory=FeatureImportanceMethodConfig)
    shap: SHAPMethodConfig = Field(default_factory=SHAPMethodConfig)

class ExplainabilityConfig(BaseModel):
    """Top-level explainability settings used during post-training analysis."""

    enabled: bool = True
    top_k: int = 20
    methods: ExplainabilityMethodsConfig = Field(default_factory=ExplainabilityMethodsConfig)

DATA_TYPE = Literal["tabular", "time-series"]

class ModelSpecsLineageConfig(BaseModel):
    """Lineage metadata for model specification creation."""

    created_by: str
    created_at: datetime

class MetaConfig(BaseModel):
    """Runtime metadata attached during config loading/validation."""

    model_config = ConfigDict(extra="allow")
    sources: Optional[dict[str, Any]] = None
    env: Optional[str] = None
    best_params_path: Optional[str] = None
    validation_status: Optional[str] = None
    validation_errors: Optional[list[Any]] = None
    config_hash: Optional[str] = None

class ModelSpecs(BaseModel):
    """Canonical model specification shared by search and training schemas."""

    problem: str
    segment: SegmentConfig
    version: str
    task: TaskConfig
    target: TargetConfig
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    min_rows: int = 10000
    split: SplitConfig
    algorithm: AlgorithmConfig
    model_class: str
    pipeline: PipelineConfig
    scoring: ScoringConfig
    class_weighting: ClassWeightingConfig = Field(default_factory=ClassWeightingConfig)
    feature_store: FeatureStoreConfig
    explainability: ExplainabilityConfig = Field(default_factory=ExplainabilityConfig)
    data_type: DATA_TYPE
    model_specs_lineage: ModelSpecsLineageConfig
    meta: MetaConfig = Field(default_factory=MetaConfig, alias="_meta")

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    @model_validator(mode="after")
    def validate_task_target_consistency(self):
        """Enforce consistency between task type and target class metadata.

        Returns:
            ModelSpecs: Validated model specs instance.
        """

        # --- Classification rules ---
        if self.task.type == TaskType.classification:
            if self.target.classes is None:
                msg = "Classes must be provided for classification tasks."
                logger.error(msg)
                raise ConfigError(msg)

            if self.target.classes.count < 2:
                msg = (
                    f"Classes count must be at least 2 for classification tasks, "
                    f"got {self.target.classes.count}."
                )
                logger.error(msg)
                raise ConfigError(msg)

        # --- Non-classification rules ---
        else:
            if self.target.classes is not None:
                msg = (
                    f"Classes should not be provided for task type "
                    f"'{self.task.type}'."
                )
                logger.error(msg)
                raise ConfigError(msg)

        return self
    
    # Validate that target.transform.enabled is False and target.transform.type is None for non-regression tasks, while for regression tasks, if target.transform.enabled is True, then target.transform.type must be specified
    @model_validator(mode="after")
    def validate_target_transform_consistency(cls, v):
        """Validate target transformation compatibility with task type.

        Args:
            v: Candidate model specs instance.

        Returns:
            ModelSpecs: Validated model specs instance.
        """

        if v.task.type != TaskType.regression:
            if v.target.transform.enabled:
                msg = f"Target transformation is only applicable for regression tasks. Found enabled for task type '{v.task.type}'."
                logger.error(msg)
                raise ConfigError(msg)
            if v.target.transform.type is not None:
                msg = f"Target transformation type should be None for non-regression tasks. Found '{v.target.transform.type}' for task type '{v.task.type}'."
                logger.error(msg)
                raise ConfigError(msg)
        else:
            if v.target.transform.enabled and v.target.transform.type is None:
                msg = "Target transformation type must be specified when target transformation is enabled for regression tasks."
                logger.error(msg)
                raise ConfigError(msg)
        return v
    
    # validate that class_weighting is off for non-classification tasks
    @model_validator(mode="after")
    def validate_class_weighting_consistency(cls, v):
        """Ensure class weighting is only enabled for classification tasks.

        Args:
            v: Candidate model specs instance.

        Returns:
            ModelSpecs: Validated model specs instance.
        """

        if v.task.type != TaskType.classification and v.class_weighting.policy != "off":
            msg = f"Class weighting is only applicable for classification tasks. Found policy '{v.class_weighting.policy}' for task type '{v.task.type}'."
            logger.error(msg)
            raise ConfigError(msg)
        return v