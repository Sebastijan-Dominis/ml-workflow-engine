import logging
logger = logging.getLogger(__name__)
import yaml
from pathlib import Path

def prepare_config(best_params):
    # Strip leading "Model__" from keys if present
    train_params = {
        (k[len("Model__"): ] if k.startswith("Model__") else k): v
        for k, v in best_params.items()
    }

    training_config = {
        "train_params": {
            **train_params
        }
    }

    return training_config

def save_training_config(cfg_model_specs, best_params):
    training_config = prepare_config(best_params)

    config_path = Path(f"configs/train/{cfg_model_specs['problem']}/{cfg_model_specs['segment']['name']}/{cfg_model_specs['version']}.yaml")
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Warn if overwriting existing config
    if config_path.exists():
        logger.warning(f"Overwriting existing training config at {config_path}")

    with open(config_path, 'w') as file:
        yaml.safe_dump(training_config, file, sort_keys=False)

    logger.info(f"Training configuration saved at {config_path}")