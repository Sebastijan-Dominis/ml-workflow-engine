"""Module for catalogs of machine learning components, such as feature operators, loss functions, model classes, and pipeline components."""

from .feature_operators_catalog import FEATURE_OPERATORS
from .io_readers_catalog import FORMAT_REGISTRY_READ
from .loss_functions_catalog import REGRESSION_LOSS_FUNCTIONS
from .model_classes_catalog import MODEL_CLASS_REGISTRY
from .model_params_catalog import MODEL_PARAM_REGISTRY
from .operator_aliases_catalog import OP_MAP
from .pipeline_components_catalog import PIPELINE_COMPONENTS

__all__ = [
    "FEATURE_OPERATORS",
    "FORMAT_REGISTRY_READ",
    "REGRESSION_LOSS_FUNCTIONS",
    "MODEL_CLASS_REGISTRY",
    "MODEL_PARAM_REGISTRY",
    "OP_MAP",
    "PIPELINE_COMPONENTS",
]