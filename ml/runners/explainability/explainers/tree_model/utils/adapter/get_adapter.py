import logging

from catboost import CatBoost

from ml.exceptions import PipelineContractError
from ml.runners.explainability.explainers.tree_model.adapters.base import \
    TreeModelAdapter
from ml.runners.explainability.explainers.tree_model.adapters.catboost import \
    CatBoostAdapter

logger = logging.getLogger(__name__)

def get_tree_model_adapter(model) -> TreeModelAdapter:

    if isinstance(model, CatBoost):
        return CatBoostAdapter(model)

    msg = f"No adapter found for model type: {type(model).__name__}"
    logger.error(msg)
    raise PipelineContractError(msg)