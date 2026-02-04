import logging
logger = logging.getLogger(__name__)
import yaml
import numpy as np
from pathlib import Path
from typing import Any

from ml.search.searchers.base import BaseSearcher
from ml.search.utils.utils import perform_randomized_search
from ml.utils.features import get_cat_features, load_X_and_y, load_schemas, validate_model_feature_pipeline_contract
from ml.search.searchers.catboost.model import (
    prepare_model,
    build_pipeline_with_model,
)
from ml.search.params.common import flatten_search_params
from ml.search.params.catboost.refinement import prepare_narrow_params
from ml.search.params.catboost.validation import validate_param_value
from ml.exceptions import (
    ConfigError,
    SearchError,
)

class SearchCatboost(BaseSearcher):
    def search(self, model_cfg: dict[str, Any]) -> dict[str, Any]:
        X_train, y_train = load_X_and_y(model_cfg, keys=["X_train", "y_train"])
        input_schema, derived_schema = load_schemas(model_cfg)

        pipeline_path = Path(f"{model_cfg['pipeline']['path']}").resolve()
        with pipeline_path.open("r") as f:
            pipeline_cfg = yaml.safe_load(f)

        cat_features = get_cat_features(input_schema, derived_schema)

        validate_model_feature_pipeline_contract(
            model_cfg,
            pipeline_cfg,
            cat_features
        )

        model_1 = prepare_model(model_cfg, "broad", cat_features)

        pipeline_1 = build_pipeline_with_model(
            pipeline_cfg=pipeline_cfg,
            input_schema=input_schema,
            derived_schema=derived_schema,
            model=model_1
        )

        broad_param_distributions_raw = model_cfg.get("search", {}).get("broad", {}).get("param_distributions", {})

        if not broad_param_distributions_raw:
            msg = f"No broad search param_distributions defined in the model config for problem={model_cfg['problem']} segment={model_cfg['segment']['name']} version={model_cfg['version']}."
            logger.error(msg)
            raise ConfigError(msg)

        broad_param_distributions = flatten_search_params(broad_param_distributions_raw)

        logger.info("Starting broad hyperparameter search | problem=%s segment=%s version=%s",
            model_cfg['problem'], model_cfg['segment']['name'], model_cfg['version'])

        logger.debug("Broad search param combinations: %d", np.prod([len(v) for v in broad_param_distributions.values()]))

        try:
            broad_result = perform_randomized_search(
                pipeline_1,
                X_train,
                y_train,
                broad_param_distributions,
                model_cfg,
                search_type="broad"
            )
        except Exception as e:
            msg = f"Broad hyperparameter search failed for problem={model_cfg['problem']} segment={model_cfg['segment']['name']} version={model_cfg['version']}: {e}"
            logger.error(msg)
            raise SearchError(msg) from e

        best_params_1 = broad_result["best_params"]

        if model_cfg.get("search", {}).get("narrow", {}).get("enabled", False) is not True:
            return {
                "best_params": best_params_1,
                "phases": {
                    "broad": broad_result
                }
            }

        narrow_param_cfg = model_cfg.get("search", {}).get("narrow", {}).get("param_configurations", {})

        if not narrow_param_cfg:
            msg = f"No narrow search param_configurations defined in the model config for problem={model_cfg['problem']} segment={model_cfg['segment']['name']} version={model_cfg['version']}."
            logger.error(msg)
            raise ConfigError(msg)

        narrow_param_distributions = prepare_narrow_params(best_params_1, narrow_param_cfg, model_cfg["search"]["hardware"]["task_type"].value)

        for param, values in narrow_param_distributions.items():
            base_param_name = param.replace("Model__", "")
            for v in values:
                validate_param_value(base_param_name, v, str(model_cfg["search"]["hardware"]["task_type"].value).upper())

        model_2 = prepare_model(model_cfg, "narrow", cat_features)

        pipeline_2 = build_pipeline_with_model(
            pipeline_cfg=pipeline_cfg,
            input_schema=input_schema,
            derived_schema=derived_schema,
            model=model_2
        )

        logger.info("Starting narrow hyperparameter search | problem=%s segment=%s version=%s",
            model_cfg['problem'], model_cfg['segment']['name'], model_cfg['version'])

        logger.debug("Narrow search param combinations: %d", np.prod([len(v) for v in narrow_param_distributions.values()]))

        try:
            narrow_result = perform_randomized_search(
                pipeline_2,
                X_train,
                y_train,
                narrow_param_distributions,
                model_cfg,
                search_type="narrow"
            )
        except Exception as e:
            msg = f"Narrow hyperparameter search failed for problem={model_cfg['problem']} segment={model_cfg['segment']['name']} version={model_cfg['version']}: {e}"
            logger.error(msg)
            raise SearchError(msg) from e

        best_params = narrow_result["best_params"]
        
        return {
            "best_params": best_params,
            "phases": {
                "broad": broad_result,
                "narrow": narrow_result
            }
        }