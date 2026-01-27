import logging
logger = logging.getLogger(__name__)
import yaml
from pathlib import Path

def prepare_config(cfg, best_params):
    pipeline_steps = cfg['pipeline_steps']
    validate_schema = pipeline_steps['validate_schema']
    fill_categorical_missing = pipeline_steps['fill_categorical_missing']
    feature_engineering = pipeline_steps['feature_engineering']
    feature_selection = pipeline_steps['feature_selection']

    task = cfg['task']

    if task in ['binary_classification', 'regression']:
        explainability = {
            'feature_importance_method': 'PredictionValuesChange',
            'shap_method': 'tree_explainer_mean_abs'
        }
    else:
        explainability = {}

    training_config = {
        'name': cfg['name'],
        'task': task,
        'version': cfg['version'],

        'data': {
            "components_path": cfg["artifacts"]["components_path"],
            "engineered_features": cfg["features"].get("engineered_features", []),
            "engineered_categorical_features": cfg["features"].get("engineered_categorical_features", []),
            "schema_path": cfg["features"]["schema_path"],
            "features_path": cfg["features"]["features_path"],
            "features_version": cfg["features"]["features_version"],
            "target": cfg["target"],
            "train_file": "X_train.parquet",
            "val_file": "X_val.parquet",
            "test_file": "X_test.parquet",
            "y_train": "y_train.parquet",
            "y_val": "y_val.parquet",
            "y_test": "y_test.parquet"
        },

        'model': {
            'algorithm': cfg['algorithm'],
            'params': {
                **best_params,
                **cfg["training_params"]
            },
            'class_weights': cfg.get("class_weights", None),
        },

        'pipeline': {
            'validate_schema': validate_schema,
            'fill_categorical_missing': fill_categorical_missing,
            'feature_engineering': feature_engineering,
            'feature_selection': feature_selection
        },

        'explainability': explainability
    }

    return training_config

def save_training_config(cfg, best_params):
    training_config = prepare_config(cfg, best_params)

    config_path = Path(f"ml/training/train_configs/{training_config['name']}_{training_config['version']}.yaml")
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Warn if overwriting existing config
    if config_path.exists():
        logger.warning(f"Overwriting existing training config at {config_path}")

    with open(config_path, 'w') as file:
        yaml.safe_dump(training_config, file, sort_keys=False)

    logger.info(f"Training configuration saved at {config_path}")