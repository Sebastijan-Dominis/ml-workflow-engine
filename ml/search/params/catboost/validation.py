import logging
logger = logging.getLogger(__name__)

from ml.registry.param_constraints.catboost import CATBOOST_PARAM_CONSTRAINTS


def validate_param_value(param_name: str, value, task_type: str):
    constraints = CATBOOST_PARAM_CONSTRAINTS.get(param_name)
    if not constraints:
        return

    if value is None:
        return

    if not constraints.allow_zero and value == 0:
        msg = f"{param_name} cannot be zero"
        logger.error(msg)
        raise ValueError(msg)

    if not constraints.allow_negative and value < 0:
        msg = f"{param_name} cannot be negative"
        logger.error(msg)
        raise ValueError(msg)

    if constraints.min_value is not None and value < constraints.min_value:
        msg = f"{param_name}={value} < min allowed {constraints.min_value}"
        logger.error(msg)
        raise ValueError(msg)

    if constraints.max_value is not None and value > constraints.max_value:
        msg = f"{param_name}={value} > max allowed {constraints.max_value}"
        logger.error(msg)
        raise ValueError(msg)
    
    if param_name == "border_count" and task_type == "GPU" and value not in [32, 64, 128, 254]:
        msg = f"{param_name} has to be one of [32, 64, 128, 254] for GPU task type, got {value}"
        logger.error(msg)
        raise ValueError(msg)
    
    if param_name == "colsample_bylevel" and task_type == "GPU" and value != 1.0:
        msg = f"{param_name} has to be 1.0 for GPU task type, got {value}"
        logger.error(msg)
        raise ValueError(msg)