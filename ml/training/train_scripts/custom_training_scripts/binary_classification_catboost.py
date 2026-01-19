"""Binary classification training utilities using CatBoost.

This module contains helper routines and a top-level training function
used to train binary classification models with CatBoost within the
project training framework. It provides deterministic data loading,
dynamic import of model-specific pipeline components from the
``ml.components`` package, pipeline construction, model definition,
and training orchestration.

The public entrypoint is ``train_binary_classification_with_catboost`` which
returns a fitted ``sklearn.pipeline.Pipeline`` combining preprocessing
steps and the trained CatBoost model.
"""

# General imports
import logging
logger = logging.getLogger(__name__)
import pandas as pd
import importlib

from pathlib import Path
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier

def load_data(cfg: dict) -> tuple:
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

def import_components(name_and_version: str) -> tuple:
    """Dynamically import component definitions for a specific model.

    The project defines per-model pipeline components under ``ml.components``
    using a module named after the model (for example ``cancellation_v1``).
    This function imports that module and extracts a small set of
    expected attributes used to construct the preprocessing pipeline.

    Args:
        name_and_version (str): Module name under ``ml.components``.

    Returns:
        tuple: A tuple with the following items in order:
            - ``categorical_features`` (list)
            - ``required_features`` (list)
            - ``cat_features`` (list)
            - ``SchemaValidator`` (callable/class)
            - ``FillCategoricalMissing`` (callable/class)
            - ``FeatureEngineer`` (callable/class)
            - ``FeatureSelector`` (callable/class)

    Raises:
        ImportError: If the module cannot be imported.
        AttributeError: If the module does not expose the required attributes.
    """

    try:
        module = importlib.import_module(f"ml.components.{name_and_version}")
    except ImportError:
        logger.exception(f"Error importing module ml.components.{name_and_version}")
        raise

    required_attributes = [
        "categorical_features",
        "required_features",
        "cat_features",
        "SchemaValidator",
        "FillCategoricalMissing",
        "FeatureEngineer",
        "FeatureSelector",
    ]

    missing = [a for a in required_attributes if not hasattr(module, a)]

    if missing:
        raise AttributeError(
            f"Missing attributes in module ml.components.{name_and_version}: {missing}"
        )

    return (
        module.categorical_features,
        module.required_features,
        module.cat_features,
        module.SchemaValidator,
        module.FillCategoricalMissing,
        module.FeatureEngineer,
        module.FeatureSelector,
    )

def define_model(cfg: dict, cat_features: list) -> CatBoostClassifier:
    """Construct a CatBoostClassifier from configuration.

    Args:
        cfg (dict): Validated configuration dictionary with ``model.params``.
        cat_features (list): List of categorical feature names or indices
            to pass to CatBoost.

    Returns:
        CatBoostClassifier: Instantiated (but unfitted) CatBoost model.
    """

    # The model uses CPU for compatibility in production environments
    # Extract only non-None parameters to allow defaults
    params = {
        k: v
        for k, v in cfg["model"]["params"].items()
        if v is not None
    }

    model_class = CatBoostClassifier(
        **params,
        cat_features=cat_features,
        early_stopping_rounds=100,
    )
    
    return model_class

def build_pipeline_steps(
    cfg: dict,
    SchemaValidator: type,
    FillCategoricalMissing: type,
    FeatureEngineer: type,
    FeatureSelector: type,
    model_class: CatBoostClassifier,
    required_features: list,
    categorical_features: list,
) -> list:
    """Assemble a list of (name, transformer) steps for a sklearn Pipeline.

    The function inspects pipeline flags in ``cfg['pipeline']`` and
    conditionally includes validators, missing-value fillers, feature
    engineering, and feature selection steps. The final step is the
    CatBoost model instance.

    Args:
        cfg (dict): Configuration controlling which pipeline steps to include.
        SchemaValidator, FillCategoricalMissing, FeatureEngineer, FeatureSelector:
            Classes or callables used to construct pipeline transformers.
        model_class (CatBoostClassifier): The model instance to attach.
        required_features (list): Columns required for schema validation.
        categorical_features (list): Columns treated as categorical.

    Returns:
        list: Pipeline steps as expected by ``sklearn.pipeline.Pipeline``.
    """

    steps = []
    if cfg["pipeline"]["validate_schema"]:
        steps.append(("schema_validator", SchemaValidator(required_columns=required_features)))
    if cfg["pipeline"]["fill_categorical_missing"]:
        steps.append(("fill_categorical_missing", FillCategoricalMissing(categorical_columns=categorical_features)))
    if cfg["pipeline"]["feature_engineering"]:
        steps.append(("feature_engineering", FeatureEngineer()))
    if cfg["pipeline"]["feature_selection"]:
        # Make sure FeatureEngineer contains created_columns attribute
        steps.append(("feature_selector", FeatureSelector(selected_columns=required_features + FeatureEngineer.created_columns)))

    steps.append(("model", model_class))

    return steps

def train_model(
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

def train_binary_classification_with_catboost(name_and_version: str, cfg: dict) -> Pipeline:
    """Train a binary classification model using CatBoost and project components.

    This is the high-level routine used by the training CLI to execute a
    complete training run. It performs data loading, dynamic component
    import for model-specific preprocessing, model construction, pipeline
    assembly, training, and returns a fitted pipeline object.

    Args:
        name_and_version (str): Component module name under ``ml.components``.
        cfg (dict): Validated configuration dictionary.

    Returns:
        Pipeline: A fitted sklearn Pipeline containing preprocessing and
        the trained CatBoost model.

    Raises:
        Exception: Any exception during the training process is logged and
        re-raised to signal a fatal training error.
    """

    try:
        X_train, y_train, X_val, y_val = load_data(cfg) # Load data

        (
            categorical_features,
            required_features,
            cat_features,
            SchemaValidator,
            FillCategoricalMissing,
            FeatureEngineer,
            FeatureSelector,
        ) = import_components(name_and_version) # Import components

        model_class = define_model(cfg, cat_features) # Define model

        steps = build_pipeline_steps(
            cfg,
            SchemaValidator,
            FillCategoricalMissing,
            FeatureEngineer,
            FeatureSelector,
            model_class,
            required_features,
            categorical_features,
        )

        pipeline = train_model(steps, X_train, y_train, X_val, y_val) # Train model

        logger.info(f"Model {cfg['name']}_{cfg['version']} trained successfully.") # Log success

        return pipeline # Return fitted pipeline
    except Exception:
        logger.exception(f"Training failed for model {cfg['name']}_{cfg['version']}") # Log failure
        raise