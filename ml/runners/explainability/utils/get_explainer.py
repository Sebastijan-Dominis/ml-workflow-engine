import logging

from ml.exceptions import PipelineContractError
from ml.registry.explainability_registry import EXPLAINERS
from ml.runners.explainability.explainers.base import Explainer

logger = logging.getLogger(__name__)

def get_explainer(key: str) -> Explainer:
    explainer_cls = EXPLAINERS.get(key)

    if not explainer_cls:
        msg = f"No explainer found for algorithm '{key}'."
        logger.error(msg)
        raise PipelineContractError(msg)
    
    evaluator = explainer_cls()

    logger.debug(
        "Using explainer %s for algorithm=%s",
        evaluator.__class__.__name__,
        key,
    )

    return evaluator