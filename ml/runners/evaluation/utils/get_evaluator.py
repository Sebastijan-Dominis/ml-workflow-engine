import logging

from ml.exceptions import PipelineContractError
from ml.registry.evaluation_registry import EVALUATORS
from ml.runners.evaluation.evaluators.base import Evaluator

logger = logging.getLogger(__name__)

def get_evaluator(key: str) -> Evaluator:
    evaluator_cls = EVALUATORS.get(key)

    if not evaluator_cls:
        msg = f"No evaluator found for algorithm '{key}'."
        logger.error(msg)
        raise PipelineContractError(msg)
    
    evaluator = evaluator_cls()

    logger.debug(
        "Using evaluator %s for algorithm=%s",
        evaluator.__class__.__name__,
        key,
    )

    return evaluator