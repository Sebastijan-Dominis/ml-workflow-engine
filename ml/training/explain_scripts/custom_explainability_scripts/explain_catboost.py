import importlib
import joblib
import pandas as pd
import numpy as np
import shap

from pathlib import Path

# -------------------------------------------
# Helper function to import custom components
# -------------------------------------------
def import_components(name_and_version):
    # Dynamically import the module for the given model
    importlib.import_module(f"ml.components.{name_and_version}")

# -------------------------------------------
# Helper function to check key presence in model configs
# ------------------------------------------
def check_key_presence(model_configs):
    required_keys = ["explainability", "features", "name", "version"]
    for k in required_keys:
        if k not in model_configs:
            raise KeyError(f"Missing required config key: {k}")


# ------------------------------------------
# Helper function to load the trained model and pipeline
# ------------------------------------------
def get_pipeline_and_model(name_and_version):
    # Import custom components to ensure deserialization works
    import_components(name_and_version)

    # Load the trained pipeline
    pipeline = joblib.load(f"ml/models/trained/{name_and_version}.joblib")
    
    # Extract the model from the pipeline
    model = pipeline.named_steps["model"]

    # Return the model and pipeline
    return model, pipeline

# ------------------------------------------
# Helper function to inspect the pipeline
# ------------------------------------------
def inspect_pipeline(pipeline):
    if not hasattr(pipeline, "named_steps"):
        raise TypeError("Pipeline must be a sklearn Pipeline")

    if list(pipeline.named_steps.keys())[-1] != "model":
        raise ValueError("Expected 'model' to be the final pipeline step")

    if not hasattr(pipeline[:-1], "transform"):
        raise ValueError("Pipeline preprocessing steps must be transformable")

# ------------------------------------------
# Helper function to get feature names from the pipeline
# ------------------------------------------
def get_feature_names(pipeline, X):
    X_transformed = pipeline[:-1].transform(X)

    if hasattr(X_transformed, "columns"):
        return X_transformed.columns.to_numpy()

    raise ValueError(
        "Transformed data has no column names. "
        "Feature names must be provided by the transformer."
    )

# ------------------------------------------
# Helper function to get feature importances from the model
# ------------------------------------------
def get_feature_importances(feature_names, model, model_configs):
    # Get feature importances
    try:
        importances = model.get_feature_importance(type=model_configs["explainability"]["feature_importance_method"])
    except KeyError as ke:
        raise KeyError(f"Feature importance method either not specified or invalid: {ke}")
    except Exception as e:    
        raise RuntimeError(f"Error getting feature importances: {e}")

    # Validate lengths
    if len(feature_names) != len(importances):
        raise ValueError("Mismatch between feature names and importances")

    # Create a dictionary mapping feature names to their importances, for top 20 features
    df_imp_top_20 = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False).head(20)
    
    # Return the top 20 feature importances
    return df_imp_top_20

# -------------------------------------------
# Helper function to save feature importances to a CSV file
# -------------------------------------------
def save_feature_importances(df, name_and_version):
    # Create directory if it doesn't exist
    Path(f"ml/models/explainability/{name_and_version}").mkdir(parents=True, exist_ok=True)

    # Save the feature importances to a CSV file
    df.to_csv(f"ml/models/explainability/{name_and_version}/feature_importances.csv", index=False)

    # Print success message
    print("Feature importances saved successfully.")

# -------------------------------------------
# Helper function to load test data
# -------------------------------------------
def get_test_data(model_configs):
    features_path = Path(model_configs["features"]["path"])
    X_test = pd.read_parquet(features_path/"X_test.parquet")
    return X_test

# ------------------------------------------
# Helper function to get SHAP importances
# ------------------------------------------
def get_shap_importances(feature_names, model, pipeline, X_test, model_configs):
    """
    Calculates SHAP importances for a trained CatBoost model.
    """
    # Transform the input data using the pipeline
    X_test_transformed = pipeline[:-1].transform(X_test)

    # Sample a subset of the test data for SHAP calculations
    n = min(1000, X_test_transformed.shape[0])
    rng = np.random.default_rng(42)
    idx = rng.choice(X_test_transformed.shape[0], size=n, replace=False)

    if not hasattr(X_test_transformed, "iloc"):
        raise TypeError("Transformed data must be a pandas DataFrame for SHAP analysis")
    X_test_sample = X_test_transformed.iloc[idx]

    # Create SHAP TreeExplainer for the model
    if model_configs["explainability"]["shap_method"] == "tree_explainer_mean_abs":
        explainer = shap.TreeExplainer(
            model,
            feature_perturbation="tree_path_dependent",
            model_output="raw"
        )

        # Calculate SHAP values
        shap_values = explainer.shap_values(X_test_sample)

        # Compute mean absolute SHAP values
        if isinstance(shap_values, list):
            shap_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)
        else:
            shap_values = np.abs(shap_values)

        # Compute mean absolute SHAP values for each feature
        importances = shap_values.mean(axis=0)

        # Validate lengths
        if len(feature_names) != len(importances):
            raise ValueError("Mismatch between feature names and importances")

        # Create a dictionary mapping feature names to their global SHAP importance, for top 20 features
        top_20_shap_importances = pd.DataFrame({
            'feature': feature_names,
            'mean_abs_shap': importances
        }).sort_values(by='mean_abs_shap', ascending=False).head(20)

        # Return SHAP importances
        return top_20_shap_importances
    
    else:
        raise ValueError(
            f"Unsupported SHAP method: {model_configs['explainability']['shap_method']}"
        )

# ------------------------------------------
# Helper function to save SHAP importances to a CSV file
# ------------------------------------------
def save_shap_importances(df, name_and_version):
    """
    Saves the SHAP importances DataFrame to a CSV file.
    """
    # Create directory if it doesn't exist
    Path(f"ml/models/explainability/{name_and_version}").mkdir(parents=True, exist_ok=True)

    # Save the SHAP importances to a CSV file
    df.to_csv(f"ml/models/explainability/{name_and_version}/shap_importances.csv", index=False)

    # Print success message
    print("SHAP importances saved successfully.")

# ------------------------------------------
# Main explainability function for CatBoost models
# ------------------------------------------
def explain_catboost(model_configs):
    """
    Runs global explainability for a trained CatBoost model.
    Produces feature importance and SHAP-based global explanations.
    """
    # Step 1 - Extract name and version
    name_and_version = f"{model_configs['name']}_{model_configs['version']}"

    # Step 2 - Check key presence
    check_key_presence(model_configs)

    # Step 3 - Load the trained model and pipeline
    model, pipeline = get_pipeline_and_model(name_and_version)

    # Step 4 - Inspect the pipeline
    inspect_pipeline(pipeline)

    # Step 5 - Load test data
    X_test = get_test_data(model_configs)

    # Step 6 - Get feature names
    feature_names = get_feature_names(pipeline, X_test)

    # Step 7 - Get feature importances
    top_20_feature_importances = get_feature_importances(feature_names, model, model_configs)

    # Step 8 - Save feature importances
    save_feature_importances(top_20_feature_importances, name_and_version)

    # Step 9 - Get SHAP importances
    top_20_shap_importances = get_shap_importances(feature_names, model, pipeline, X_test, model_configs)

    # Step 10 - Save SHAP importances
    save_shap_importances(top_20_shap_importances, name_and_version)