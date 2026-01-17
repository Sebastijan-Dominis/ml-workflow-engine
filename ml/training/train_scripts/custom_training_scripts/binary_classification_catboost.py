import pandas as pd
import importlib

from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier

# ------------------------------------------
# Helper function to load data
# ------------------------------------------
def load_data(cfg):
    feature_path = cfg["data"]["features_path"]

    X_train = pd.read_parquet(feature_path + cfg["data"]["train_file"])
    X_val   = pd.read_parquet(feature_path + cfg["data"]["val_file"])

    y_train = pd.read_parquet(feature_path + cfg["data"]["y_train"])
    y_val   = pd.read_parquet(feature_path + cfg["data"]["y_val"])

    return X_train, y_train, X_val, y_val

# -------------------------------------------
# Helper function to import custom components
# -------------------------------------------
def import_components(name_and_version):
    module = importlib.import_module(f"ml.components.{name_and_version}")

    categorical_features = module.categorical_features
    required_features = module.required_features
    cat_features = module.cat_features
    SchemaValidator = module.SchemaValidator
    FillCategoricalMissing = module.FillCategoricalMissing
    FeatureEngineer = module.FeatureEngineer
    FeatureSelector = module.FeatureSelector

    return (categorical_features, required_features, cat_features,
            SchemaValidator, FillCategoricalMissing, FeatureEngineer, FeatureSelector)

# ---------------------------------------------------
# Helper function to define the model
# ---------------------------------------------------
def define_model(cfg, cat_features):
    # The model uses CPU for compatibility in production environments
    model_class = CatBoostClassifier(
        **cfg["model"]["params"],
        cat_features=cat_features,
        early_stopping_rounds=100,
    )
    return model_class

# ---------------------------------------------------
# Helper function to build pipeline steps
# ---------------------------------------------------
def build_pipeline_steps(cfg,
                         SchemaValidator,
                         FillCategoricalMissing,
                         FeatureEngineer,
                         FeatureSelector,
                         model_class,
                         required_features,
                         categorical_features):
    steps = []
    if cfg["pipeline"]["validate_schema"]:
        steps.append(("schema_validator", SchemaValidator(required_columns=required_features)))
    if cfg["pipeline"]["fill_categorical_missing"]:
        steps.append(("fill_categorical_missing", FillCategoricalMissing(categorical_columns=categorical_features)))
    if cfg["pipeline"]["feature_engineering"]:
        steps.append(("feature_engineering", FeatureEngineer()))
    if cfg["pipeline"]["feature_selection"]:
        steps.append(("feature_selector", FeatureSelector(selected_columns=required_features + FeatureEngineer.created_columns)))

    steps.append(("model", model_class))

    return steps

# ---------------------------------------------------
# Helper function to train and fit the model, create pipeline
# ---------------------------------------------------
def train_model(steps, X_train, y_train, X_val, y_val):
    preprocessing_pipeline = Pipeline(steps[:-1])
    model = steps[-1][1]

    X_train_processed = preprocessing_pipeline.fit_transform(X_train, y_train)
    X_val_processed = preprocessing_pipeline.transform(X_val)

    model.fit(
        X_train_processed, y_train,
        eval_set=(X_val_processed, y_val),
        use_best_model=True
    )

    pipeline = Pipeline(steps=steps[:-1] + [("model", model)])

    return pipeline

# ---------------------------------------------------
# Main training function for binary classification with CatBoost
# ---------------------------------------------------
def train_binary_classification_with_catboost(name_and_version, cfg):
    # Step 1 - Load frozen features
    X_train, y_train, X_val, y_val = load_data(cfg)

    # Step 2 - Import custom pipeline components
    (categorical_features,
     required_features,
     cat_features,
     SchemaValidator,
     FillCategoricalMissing,
     FeatureEngineer,
     FeatureSelector) = import_components(name_and_version)

    # Step 3 - Define the model
    model_class = define_model(cfg, cat_features)

    # Step 4 - Build pipeline steps
    steps = build_pipeline_steps(
        cfg,
        SchemaValidator,
        FillCategoricalMissing,
        FeatureEngineer,
        FeatureSelector,
        model_class,
        required_features,
        categorical_features
    )

    # Step 5 - Train and fit the model, create pipeline
    pipeline = train_model(steps, X_train, y_train, X_val, y_val)

    # Step 6 - Print success message
    print(f"Model {cfg['name']}_{cfg['version']} trained and saved successfully.")

    # Step 7 - Return the trained pipeline
    return pipeline