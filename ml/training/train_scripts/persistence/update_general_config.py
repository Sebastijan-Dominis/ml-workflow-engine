import yaml
from pathlib import Path

def update_general_config(cfg):
    model_key = f"{cfg['name']}_{cfg['version']}"

    # Step 1 - Prepare general config structure
    general_config = {
        model_key: {
            "name": cfg["name"],
            "task": cfg["task"],
            "target": cfg["data"]["target"],
            "algorithm": cfg["model"]["algorithm"],
            "features": {
                "version": cfg["version"],
                "path": cfg["data"]["features_path"],
                "schema": cfg["data"]["features_path"] + "schema.csv"
            },
            "artifacts": {
                "model": f"ml/models/trained/{cfg['name']}_{cfg['version']}.joblib",
                "metadata": f"ml/models/metadata/{cfg['name']}_{cfg['version']}.json",
                "feature_importance": f"ml/models/explainability/{cfg['name']}_{cfg['version']}/feature_importance.csv",
                "shap": f"ml/models/explainability/{cfg['name']}_{cfg['version']}/shap_values.parquet"
            },
            "explainability": {
                "feature_importance_method": cfg["explainability"]["feature_importance_method"],
                "shap": cfg["explainability"]["shap"]
            },
            "threshold": cfg["model"].get("threshold", 0.5)
        }
    }

    # Step 2 - Ensure directories exist
    Path(f"ml/models/trained").mkdir(parents=True, exist_ok=True)
    Path(f"ml/models/explainability/{cfg['name']}_{cfg['version']}").mkdir(parents=True, exist_ok=True)
    Path(f"ml/models/metadata").mkdir(parents=True, exist_ok=True)

    # Step 3 - Load existing general config
    general_config_path = Path("configs/models.yaml")
    general_config_path.parent.mkdir(parents=True, exist_ok=True)

    if general_config_path.exists():
        with open(general_config_path) as f:
            existing = yaml.safe_load(f) or {}
    else:
        existing = {}

    # Step 4 - Update existing config with new model info
    existing.setdefault(model_key, {}).update(general_config[model_key])

    with open(general_config_path, "w") as f:
        yaml.safe_dump(existing, f, sort_keys=False)

    # Step 5 - Print success message
    print(f"General config successfully updated with model {model_key}.")