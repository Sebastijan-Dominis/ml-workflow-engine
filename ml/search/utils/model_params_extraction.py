"""Helpers for extracting model-specific parameters from pipeline search output."""

def extract_model_params(best_params):
    """Extract estimator-level parameters from pipeline-prefixed best params.

    Args:
        best_params: Mapping of best parameters from a pipeline search.

    Returns:
        Parameter mapping containing only ``Model__`` entries without prefix.
    """

    return {
        k.split("__", 1)[1]: v
        for k, v in best_params.items()
        if k.startswith("Model__")
    }
