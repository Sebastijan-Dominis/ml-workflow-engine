# ml/pipelines/builders.py
import pandas as pd
from sklearn.pipeline import Pipeline

from ml.registry import PIPELINE_COMPONENTS, FEATURE_OPERATORS


def build_pipeline(
    pipeline_cfg: dict,
    raw_schema: pd.DataFrame,
    derived_schema: pd.DataFrame,
):
    steps = []

    # ---- schema-derived feature lists ----
    raw_features = raw_schema["feature"].tolist()
    derived_features = derived_schema["feature"].tolist()
    all_features = raw_features + derived_features

    categorical_features = raw_schema.loc[
        raw_schema["dtype"].isin(["object", "string", "category"]),
        "feature",
    ].tolist()

    # derived categoricals
    derived_categoricals = derived_schema.loc[
        derived_schema["dtype"].isin(["object", "string", "category"]),
        "feature",
    ].tolist()

    # ---- build steps ----
    for step_name in pipeline_cfg["steps"]:
        Component = PIPELINE_COMPONENTS[step_name]

        if step_name == "SchemaValidator":
            steps.append(
                ("schema_validation", Component(required_features=raw_features))
            )

        elif step_name == "FillCategoricalMissing":
            steps.append(
                ("fill_categorical_missing", Component(categorical_features))
            )

        elif step_name == "FeatureEngineer":
            operators = {
                name: FEATURE_OPERATORS[name]()
                for name in derived_schema["source_operator"].unique()
            }
            steps.append(
                (
                    "feature_engineering",
                    Component(
                        derived_schema=derived_schema,
                        operators=operators,
                    ),
                )
            )

        elif step_name == "FeatureSelector":
            steps.append(
                ("feature_selection", Component(selected_features=all_features))
            )

        elif step_name == "Model":
            # model is injected later
            pass

    return Pipeline(steps)
