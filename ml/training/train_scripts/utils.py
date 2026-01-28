import logging
logger = logging.getLogger(__name__)
from catboost import CatBoostClassifier, CatBoostRegressor
import pandas as pd
import importlib
from typing import Optional, List

from pathlib import Path
from sklearn.pipeline import Pipeline

def load_train_and_val_data(cfg: dict) -> tuple:
    """Load training and validation features and labels from disk.

    Args:
        cfg (dict): Configuration dictionary with keys under ``data``:
            - ``features_path``: base folder containing feature files.
            - ``train_file``, ``val_file``: parquet files for X.
            - ``y_train``, ``y_val``: parquet files for labels.

    Returns:
        tuple: ``(X_train, y_train, X_val, y_val)`` as pandas DataFrames/Series.

    Raises:
        Any exception encountered while reading files is logged and re-raised
        so callers can handle or fail the training run explicitly.
    """

    try:
        feature_path = Path(cfg["data"]["features_path"])

        X_train = pd.read_parquet(feature_path / cfg["data"]["train_file"])
        X_val = pd.read_parquet(feature_path / cfg["data"]["val_file"])

        y_train = pd.read_parquet(feature_path / cfg["data"]["y_train"])
        y_val = pd.read_parquet(feature_path / cfg["data"]["y_val"])
        return X_train, y_train, X_val, y_val
    except Exception:
        logger.exception("Error loading data")
        raise

def import_components(components_path: str, pipeline_cfg: dict) -> tuple:
    """Dynamically import component definitions for a specific model.

    The project defines per-model pipeline components under ``ml.components``
    using a module named after the model (for example ``cancellation_v1``).
    This function imports that module and extracts a small set of
    expected attributes used to construct the preprocessing pipeline.

    Args:
        components_path (str): Module path to import components from.
        pipeline_cfg (dict): Configuration dictionary for pipeline flags.
    Returns:
        tuple: A tuple with the following items in order:
            - ``SchemaValidator`` (callable/class)
            - ``FillCategoricalMissing`` (callable/class)
            - ``FeatureEngineer`` (callable/class)
            - ``FeatureSelector`` (callable/class)

    Raises:
        ImportError: If the module cannot be imported.
        AttributeError: If the module does not expose the required attributes.
    """

    try:
        module = importlib.import_module(components_path)
    except ImportError:
        logger.exception(f"Error importing module {components_path}")
        raise

    required_attributes = []

    if pipeline_cfg.get("validate_schema", False):
        required_attributes.append("SchemaValidator")
    if pipeline_cfg.get("fill_categorical_missing", False):
        required_attributes.append("FillCategoricalMissing")
    if pipeline_cfg.get("feature_engineering", False):
        required_attributes.append("FeatureEngineer")
    if pipeline_cfg.get("feature_selection", False):
        required_attributes.append("FeatureSelector")

    missing = [a for a in required_attributes if not hasattr(module, a)]

    if missing:
        raise AttributeError(
            f"Missing attributes in module {components_path}: {missing}"
        )

    return tuple(getattr(module, attr) for attr in required_attributes)

def get_features_from_schema(cfg: dict, schema_path: str) -> tuple:
    schema = pd.read_csv(Path(schema_path))

    categorical_features = schema.loc[schema["dtype"].isin(['object', 'category', 'string']), "feature"].tolist()

    numerical_features = schema.loc[schema["dtype"].isin(['int64', 'float64', 'int32', 'float32', 'int16', 'int8']), "feature"].tolist()

    required_features = categorical_features + numerical_features

    cat_features = categorical_features + cfg["created_cat_features"]

    return categorical_features, required_features, cat_features

def define_catboost_model(cfg: dict, cat_features: List[str]) -> CatBoostClassifier | CatBoostRegressor:
    from catboost import CatBoostClassifier, CatBoostRegressor

    params = {
        k: v
        for k, v in cfg["model"]["params"].items()
        if v is not None
    }

    # Extend these registers as needed
    SUPPORTED_CLASSIFICATION_TASKS = ["binary_classification"]
    SUPPORTED_REGRESSION_TASKS = ["regression"]

    if cfg["model"]["task"] in SUPPORTED_CLASSIFICATION_TASKS:
        model_class = CatBoostClassifier(
            **params,
            cat_features=cat_features,
            early_stopping_rounds=100,
        )
    elif cfg["model"]["task"] in SUPPORTED_REGRESSION_TASKS:
        model_class = CatBoostRegressor(
            **params,
            cat_features=cat_features,
            early_stopping_rounds=100,
        )
    else:
        raise ValueError(f"Unsupported model type: {cfg['model']['task']}")

    return model_class

def train_catboost_model(
    steps: list,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
) -> Pipeline:
    """Fit preprocessing steps and the CatBoost model, returning a Pipeline.

    The function separates preprocessing steps from the final model, fits
    the preprocessing pipeline on the training data, transforms both train
    and validation sets, and fits the CatBoost model using the transformed
    data and validation set for early stopping.

    Args:
        steps (list): Pipeline steps where the last element is the model.
        X_train, y_train, X_val, y_val: Training and validation data.

    Returns:
        Pipeline: A fitted ``sklearn.pipeline.Pipeline`` combining preprocessing
        and the trained model.
    """

    preprocessing_pipeline = Pipeline(steps[:-1])
    model = steps[-1][1]

    X_train_processed = preprocessing_pipeline.fit_transform(X_train, y_train)
    X_val_processed = preprocessing_pipeline.transform(X_val)

    model.fit(
        X_train_processed,
        y_train,
        eval_set=(X_val_processed, y_val),
        use_best_model=True,
    )

    pipeline = Pipeline(steps=steps[:-1] + [("model", model)])

    return pipeline