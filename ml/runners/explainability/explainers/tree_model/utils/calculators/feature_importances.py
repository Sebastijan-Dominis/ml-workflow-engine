import logging

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.pipeline import Pipeline

from ml.config.validation_schemas.model_cfg import TrainModelConfig
from ml.exceptions import ExplainabilityError, PipelineContractError
from ml.runners.explainability.explainers.tree_model.adapters.base import \
    TreeModelAdapter
from ml.runners.explainability.explainers.tree_model.utils.validators.validate_lengths import \
    validate_lengths

logger = logging.getLogger(__name__)

def get_feature_importances(
    *, 
    feature_names: NDArray[np.str_], 
    adapter: TreeModelAdapter,
    pipeline: Pipeline, 
    model_cfg: TrainModelConfig, 
    top_k: int
) -> pd.DataFrame | None:
    type_param = model_cfg.explainability.methods.feature_importances.type
    model = pipeline[-1]
    try:
        if  model_cfg.explainability.methods.feature_importances.enabled:
            logger.info(f"Computing feature importances using feature importances method: '{type_param}'...")
            importances = adapter.compute_feature_importances(importance_type=type_param)
        else:
            msg = "Feature importance method is not enabled in the configuration. Skipping feature importance computation."
            logger.warning(msg)
            return None
        
    except AttributeError:
        msg = f"The final estimator in the pipeline does not have 'get_feature_importance' attribute. It is of type {type(model).__name__}. Ensure that the model supports feature importance computation."
        logger.error(msg)
        raise PipelineContractError(msg)
    except Exception:
        msg = f"Error retrieving feature importances from the model, using {type_param} method on a {type(model).__name__} model. Ensure that the model is properly trained and that the pipeline is correctly defined."
        logger.exception(msg)
        raise ExplainabilityError(msg)

    importances = np.asarray(importances)
    validate_lengths(feature_names, importances)

    df_imp_top_k = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False).head(top_k)

    return df_imp_top_k