import logging
logger = logging.getLogger(__name__)
import yaml
import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor
from pathlib import Path

from ml.search.utils import perform_randomized_search
from ml.features import get_cat_features, load_X_and_y, load_schemas
from ml.pipelines.builders import build_pipeline
from ml.registry import MODEL_REGISTRY

class SearchCatboost():
    def prepare_model(self, model_specs, search_configs, search_phase, cat_features):
        model = MODEL_REGISTRY[model_specs["model_class"]](
            # Basic hyperparameters
            iterations=search_configs[search_phase]["iterations"],
            task_type=search_configs["hardware"]["task_type"],           
            devices=search_configs["hardware"]["devices"],                
            verbose=100,                
            random_state=search_configs['seed'],
            cat_features=cat_features,
            class_weights=model_specs.get("class_weights", None)
        )
        return model

    def add_model_to_pipeline(self, pipeline, model):
        pipeline.steps.append(("Model", model))
        return pipeline

    # Helper function
    def refine_int(self, center, offsets, low, high):
        values = {center}
        for o in offsets:
            values.add(center + o)
            values.add(center - o)
        return sorted(v for v in values if low <= v <= high)

    # Helper function
    def refine_float_mult(self, center, factors, low, high, decimals=5):
        values = set()
        for f in factors:
            v = center * f
            if low <= v <= high:
                values.add(round(v, decimals))
        values.add(round(center, decimals))
        return sorted(values)

    def refine_border_count(self, center):
        options = [32, 64, 128, 254]
        if center in options:
            idx = options.index(center)
            refined = set()
            if idx > 0:
                refined.add(options[idx - 1])
            refined.add(center)
            if idx < len(options) - 1:
                refined.add(options[idx + 1])
            return sorted(refined)

    def prepare_narrow_search_params(self, best_params: dict, search_params: dict) -> dict:
        narrow_params = {}

        # Tree depth
        if "Model__depth" in best_params and search_params["Model__depth"]["include"]:
            narrow_params["Model__depth"] = self.refine_int(
                best_params["Model__depth"],
                offsets=search_params["Model__depth"].get("offsets", [1, 2]),
                low=search_params["Model__depth"].get("low", 2),
                high=search_params["Model__depth"].get("high", 12)
            )

        # Learning rate
        if "Model__learning_rate" in best_params and search_params["Model__learning_rate"]["include"]:
            narrow_params["Model__learning_rate"] = self.refine_float_mult(
                best_params["Model__learning_rate"],
                factors=search_params["Model__learning_rate"].get("factors", [0.7, 0.85, 1.0, 1.15, 1.3]),
                low=search_params["Model__learning_rate"].get("low", 0.003),
                high=search_params["Model__learning_rate"].get("high", 0.5),
                decimals=search_params["Model__learning_rate"].get("decimals", 5)
            )

        # L2 regularization
        if "Model__l2_leaf_reg" in best_params and search_params["Model__l2_leaf_reg"]["include"]:
            narrow_params["Model__l2_leaf_reg"] = self.refine_int(
                best_params["Model__l2_leaf_reg"],
                offsets=search_params["Model__l2_leaf_reg"].get("offsets", [1, 2, 3]),
                low=search_params["Model__l2_leaf_reg"].get("low", 1),
                high=search_params["Model__l2_leaf_reg"].get("high", 30)
            )

        # Bagging temperature
        if "Model__bagging_temperature" in best_params and search_params["Model__bagging_temperature"]["include"]:
            narrow_params["Model__bagging_temperature"] = self.refine_float_mult(
                best_params["Model__bagging_temperature"],
                factors=search_params["Model__bagging_temperature"].get("factors", [0.6, 0.8, 1.0, 1.2, 1.5]),
                low=search_params["Model__bagging_temperature"].get("low", 0.0),
                high=search_params["Model__bagging_temperature"].get("high", 5.0),
                decimals=search_params["Model__bagging_temperature"].get("decimals", 3)
            )

        # Min data in leaf
        if "Model__min_data_in_leaf" in best_params and search_params["Model__min_data_in_leaf"]["include"]:
            narrow_params["Model__min_data_in_leaf"] = self.refine_int(
                best_params["Model__min_data_in_leaf"],
                offsets=search_params["Model__min_data_in_leaf"].get("offsets", [2, 5, 10]),
                low=search_params["Model__min_data_in_leaf"].get("low", 1),
                high=search_params["Model__min_data_in_leaf"].get("high", 50)
            )

        # Random strength
        if "Model__random_strength" in best_params and search_params["Model__random_strength"]["include"]:
            narrow_params["Model__random_strength"] = self.refine_int(
                best_params["Model__random_strength"],
                offsets=search_params["Model__random_strength"].get("offsets", [1, 2, 4]),
                low=search_params["Model__random_strength"].get("low", 0),
                high=search_params["Model__random_strength"].get("high", 20)
            )

        # Freeze GPU-sensitive / discretization params
        if "Model__border_count" in best_params and search_params["Model__border_count"]["include"]:
            narrow_params["Model__border_count"] = self.refine_border_count(
                best_params["Model__border_count"]
            )

        if "Model__colsample_bylevel" in best_params and search_params["Model__colsample_bylevel"]["include"]:
            narrow_params["Model__colsample_bylevel"] = self.refine_float_mult(
                best_params["Model__colsample_bylevel"],
                factors=search_params["Model__colsample_bylevel"].get("factors", [0.9, 1.0, 1.1]),
                low=search_params["Model__colsample_bylevel"].get("low", 0.5),
                high=search_params["Model__colsample_bylevel"].get("high", 1.0),
                decimals=search_params["Model__colsample_bylevel"].get("decimals", 2)
            )

        return narrow_params

    def build_pipeline_with_model(self, pipeline_cfg, input_schema, derived_schema, model):
        pipeline = build_pipeline(pipeline_cfg, input_schema, derived_schema)

        if not isinstance(model, (CatBoostClassifier, CatBoostRegressor)):
            msg = "Defined model is not a CatBoostClassifier or CatBoostRegressor instance."
            logger.error(msg)
            raise TypeError(msg)

        return self.add_model_to_pipeline(pipeline, model)

    def search(self, model_specs: dict, search_configs: dict) -> dict:
        X_train, y_train = load_X_and_y(model_specs, keys=["X_train", "y_train"])
        input_schema, derived_schema = load_schemas(model_specs)

        pipeline_path = Path(f"configs/pipelines/{model_specs['pipeline']}.yaml")
        with pipeline_path.open("r") as f:
            pipeline_cfg = yaml.safe_load(f)

        cat_features = get_cat_features(input_schema, derived_schema)

        model_1 = self.prepare_model(model_specs, search_configs, "broad_search", cat_features)

        pipeline_1 = self.build_pipeline_with_model(
            pipeline_cfg=pipeline_cfg,
            input_schema=input_schema,
            derived_schema=derived_schema,
            model=model_1
        )

        param_distributions_1 = search_configs["broad_search"]["param_distributions"]

        logger.info("Starting broad hyperparameter search | problem=%s segment=%s version=%s",
            model_specs['problem'], model_specs['segment']['name'], model_specs['version'])

        logger.info("Broad search param combinations: %d", np.prod([len(v) for v in param_distributions_1.values()]))

        broad_result = perform_randomized_search(
            pipeline_1,
            X_train,
            y_train,
            param_distributions_1,
            search_configs,
            search_type="broad_search"
        )

        best_params_1 = broad_result["best_params"]

        if search_configs.get("narrow_search", {}).get("enabled") is not True:
            return {
                "best_params": best_params_1,
                "phases": {
                    "broad": broad_result
                }
            }

        param_distributions_2 = self.prepare_narrow_search_params(best_params_1, search_configs["narrow_search"])

        model_2 = self.prepare_model(model_specs, search_configs, "narrow_search", cat_features)

        pipeline_2 = self.build_pipeline_with_model(
            pipeline_cfg=pipeline_cfg,
            input_schema=input_schema,
            derived_schema=derived_schema,
            model=model_2
        )

        logger.info("Starting narrow hyperparameter search | problem=%s segment=%s version=%s",
            model_specs['problem'], model_specs['segment']['name'], model_specs['version'])

        logger.info("Narrow search param combinations: %d", np.prod([len(v) for v in param_distributions_2.values()]))

        narrow_result = perform_randomized_search(
            pipeline_2,
            X_train,
            y_train,
            param_distributions_2,
            search_configs,
            search_type="narrow_search"
        )

        best_params = narrow_result["best_params"]
        
        return {
            "best_params": best_params,
            "phases": {
                "broad": broad_result,
                "narrow": narrow_result
            }
        }