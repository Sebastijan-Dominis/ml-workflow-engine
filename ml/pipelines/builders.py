"""Pipeline builder utilities for assembling configured sklearn pipelines."""

import logging

import pandas as pd
from sklearn.pipeline import Pipeline

from ml.config.schemas.model_cfg import SearchModelConfig, TrainModelConfig
from ml.exceptions import ConfigError
from ml.pipelines.operator_factory import build_operators
from ml.pipelines.schema_utils import get_pipeline_features
from ml.registries.catalogs import PIPELINE_COMPONENTS

logger = logging.getLogger(__name__)
__version__ = "1.0.0"

def build_pipeline(
    *,
    model_cfg: SearchModelConfig | TrainModelConfig,
    pipeline_cfg: dict,
    input_schema: pd.DataFrame,
    derived_schema: pd.DataFrame,
) -> Pipeline:
    """Build an sklearn ``Pipeline`` from config and schema metadata.

    Args:
        model_cfg: Validated search or training model config.
        pipeline_cfg: dict with keys 'steps' and optional 'assumptions'.
        input_schema: DataFrame with columns 'feature' and 'dtype' for raw inputs.
        derived_schema: DataFrame with columns 'feature' and 'source_operator' for engineered features.

    Returns:
        Pipeline: Configured sklearn pipeline instance.
    """
    steps = []

    # ---- schema-derived feature lists ----
    features = get_pipeline_features(
        model_cfg=model_cfg,
        input_schema=input_schema,
        derived_schema=derived_schema
    )

    input_features = features.input_features
    selected_features = features.selected_features
    categorical_features = features.categorical_features

    # ---- feature operators ----
    operators = build_operators(derived_schema)

    # ---- step handler mapping ----
    STEP_HANDLERS = {
        "SchemaValidator": lambda Component: Component(required_features=input_features),
        "FillCategoricalMissing": lambda Component: Component(categorical_features=categorical_features),
        "FeatureEngineer": lambda Component: Component(
            derived_schema=derived_schema,
            operators=operators,
        ),
        "FeatureSelector": lambda Component: Component(selected_features=selected_features),
    }

    # ---- build steps from config ----
    for step_name in pipeline_cfg.get("steps", []):
        if step_name == "Model":
            logger.debug("Skipping Model step; model should be injected later")
            continue

        if step_name not in PIPELINE_COMPONENTS:
            msg = f"Unknown pipeline step: {step_name}"
            logger.error(msg)
            raise ConfigError(msg)

        Component = PIPELINE_COMPONENTS[step_name]
        step_instance = STEP_HANDLERS[step_name](Component)
        steps.append((step_name.lower(), step_instance))
        logger.debug(f"Added pipeline step: {step_name}")

    return Pipeline(steps)
