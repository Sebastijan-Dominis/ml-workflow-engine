import logging
from pathlib import Path

from ml.config.validation_schemas.model_cfg import TrainModelConfig
from ml.exceptions import ConfigError
from ml.registry.model_param_registry import MODEL_PARAM_REGISTRY
from ml.utils.loaders import load_json

logger = logging.getLogger(__name__)

def validate_allowed_params(model_cfg: TrainModelConfig, experiment_dir: Path) -> None:
    allowed_params = MODEL_PARAM_REGISTRY[model_cfg.algorithm.value.lower()]
    metadata = load_json(experiment_dir / "experiment.json")
    best_model_params = metadata.get("best_model_params", {})
    unknown = set(best_model_params.keys()) - set(allowed_params)
    if unknown:
        msg = f"The following parameters are not allowed for algorithm {model_cfg.algorithm.value}: {unknown}"
        logger.error(msg)
        raise ConfigError(msg)
    
    logger.debug(f"All parameters in best_model_params are allowed for algorithm {model_cfg.algorithm.value}.")