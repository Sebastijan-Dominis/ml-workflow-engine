"""Registry mapping algorithm families to explainability implementations."""

from ml.runners.explainability.explainers.tree_model.tree_model import \
    ExplainTreeModel

EXPLAINERS = {
    "catboost": ExplainTreeModel
}