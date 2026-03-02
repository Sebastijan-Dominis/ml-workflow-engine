"""Adapter factory for tree-model explainability backends."""

import logging

from catboost import CatBoost

from ml.exceptions import PipelineContractError
from ml.runners.explainability.explainers.tree_model.adapters.base import \
    TreeModelAdapter
from ml.runners.explainability.explainers.tree_model.adapters.catboost import \
    CatBoostAdapter

logger = logging.getLogger(__name__)

def get_tree_model_adapter(model) -> TreeModelAdapter:
    """Resolve and instantiate tree-model adapter for the provided model.

    Args:
        model: Trained tree-based model instance requiring an explainability adapter.

    Returns:
        Adapter implementation compatible with the model type.
    """

    if isinstance(model, CatBoost):
        return CatBoostAdapter(model)

    msg = f"No adapter found for model type: {type(model).__name__}"
    logger.error(msg)
    raise PipelineContractError(msg)