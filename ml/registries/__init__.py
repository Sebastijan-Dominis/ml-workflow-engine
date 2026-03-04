"""This module serves as a central registry for various components used in the machine learning pipeline. It includes catalogs of feature operators, model classes, and loss functions, as well as factories for creating evaluators, explainers, freeze strategies, searchers, target strategies, and trainers. By centralizing these components, we can easily manage and extend the functionality of our machine learning framework."""

from .catalogs import (
    FEATURE_OPERATORS,
    MODEL_CLASS_REGISTRY,
    MODEL_PARAM_REGISTRY,
    OP_MAP,
    PIPELINE_COMPONENTS,
    REGRESSION_LOSS_FUNCTIONS,
)
from .factories import (
    EVALUATORS,
    EXPLAINERS,
    SEARCHERS,
    TRAINERS,
)

__all__ = [
    "FEATURE_OPERATORS",
    "MODEL_CLASS_REGISTRY",
    "MODEL_PARAM_REGISTRY",
    "OP_MAP",
    "PIPELINE_COMPONENTS",
    "REGRESSION_LOSS_FUNCTIONS",
    "EVALUATORS",
    "EXPLAINERS",
    "SEARCHERS",
    "TRAINERS",
]
