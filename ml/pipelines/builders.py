import logging
from typing import Dict

import pandas as pd
from sklearn.pipeline import Pipeline

from ml.pipelines import operator_factory, schema_utils
from ml.registry.pipeline_components import PIPELINE_COMPONENTS

logger = logging.getLogger(__name__)
__version__ = "1.0.0"

def build_pipeline(
    pipeline_cfg: Dict,
    input_schema: pd.DataFrame,
    derived_schema: pd.DataFrame,
) -> Pipeline:
    """
    Build an sklearn Pipeline from a config dictionary.

    Args:
        pipeline_cfg: dict with keys 'steps' and optional 'assumptions'.
        input_schema: DataFrame with columns 'feature' and 'dtype' for raw inputs.
        derived_schema: DataFrame with columns 'feature' and 'source_operator' for engineered features.

    Returns:
        sklearn.pipeline.Pipeline instance with configured steps.
    """
    steps = []

    # ---- schema-derived feature lists ----
    raw_features, _, all_features = schema_utils.get_raw_and_derived_features(
        input_schema, derived_schema
    )

    categorical_features = schema_utils.get_categorical_features(input_schema)

    # ---- feature operators ----
    operators = operator_factory.build_operators(derived_schema)

    # ---- step handler mapping ----
    STEP_HANDLERS = {
        "SchemaValidator": lambda Component: Component(required_features=raw_features),
        "FillCategoricalMissing": lambda Component: Component(categorical_features),
        "FeatureEngineer": lambda Component: Component(
            derived_schema=derived_schema,
            operators=operators,
        ),
        "FeatureSelector": lambda Component: Component(selected_features=all_features),
    }

    # ---- build steps from config ----
    for step_name in pipeline_cfg.get("steps", []):
        if step_name == "Model":
            logger.info("Skipping Model step; model should be injected later")
            continue

        if step_name not in PIPELINE_COMPONENTS:
            msg = f"Unknown pipeline step: {step_name}"
            logger.error(msg)
            raise ValueError(msg)

        Component = PIPELINE_COMPONENTS[step_name]
        step_instance = STEP_HANDLERS[step_name](Component)
        steps.append((step_name.lower(), step_instance))
        logger.debug(f"Added pipeline step: {step_name}")

    return Pipeline(steps)
