import sys
import yaml

from pathlib import Path

def save_training_config(name, version, features_path, features_version, algorithm, clean_params, iterations, threshold, pipeline_steps):
    validate_schema = pipeline_steps['validate_schema']
    fill_categorical_missing = pipeline_steps['fill_categorical_missing']
    feature_engineering = pipeline_steps['feature_engineering']
    feature_selection = pipeline_steps['feature_selection']

    training_config = {
        'name': name, # e.g., "cancellation_city_hotel_online_ta", "cancellation_global"
        'task': 'binary_classification',
        'version': version, # e.g., "v1", "v2"

        'data': {
            "features_path": features_path, # e.g., "data/features/cancellation/global/v1/", "data/features/cancellation/hotel_market_segment/city_hotel_online_ta_v1"
            "features_version": features_version, # e.g., "v1", "v2"
            "target": "is_canceled",
            "train_file": "X_train.parquet",
            "val_file": "X_val.parquet",
            "test_file": "X_test.parquet",
            "y_train": "y_train.parquet",
            "y_val": "y_val.parquet",
            "y_test": "y_test.parquet"
        },

        'model': {
            'algorithm': algorithm, # e.g., "catboost"
            'params': {
                **clean_params, # cleaned hyperparameters from search
                'iterations': iterations, # increase iterations for final training
                'task_type': 'CPU',    # switch to CPU for final model
                'random_state': 42, # reproducibility
                'verbose': 100 # print progress every 100 iterations
            },
            'threshold': threshold
        },

        'pipeline': {
            'validate_schema': validate_schema,
            'fill_categorical_missing': fill_categorical_missing,
            'feature_engineering': feature_engineering,
            'feature_selection': feature_selection
        },

        'explainability': {
            'feature_importance_method': 'PredictionValuesChange',
            'shap_method': 'tree_explainer_mean_abs'
        }
    }

    config_path = Path(f"../../../../ml/training/train_configs/{training_config['name']}_{training_config['version']}.yaml")
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, 'w') as file:
        yaml.safe_dump(training_config, file, sort_keys=False)