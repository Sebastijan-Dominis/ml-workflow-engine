import logging
logger = logging.getLogger(__name__)
from ml.validation_schemas.search_cfg import BroadParamDistributions
from ml.exceptions import ConfigError

def flatten_search_params(search_params: dict) -> dict[str, list]:
    try:
        obj = BroadParamDistributions(
            model=search_params.get("model", {}),
            ensemble=search_params.get("ensemble", {})
        )
        return obj.to_flat_dict()
    except Exception as e:
        msg = f"Error flattening search parameters."
        logger.exception(msg)
        raise ConfigError(msg) from e