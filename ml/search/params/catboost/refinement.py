from ml.search.params.utils import (
    get_default_float_params,
    get_default_int_params,
)

from ml.search.params.refiners import (
    refine_int,
    refine_float_mult,
    refine_border_count,
)

def prepare_narrow_params(best_params: dict, narrow_params_cfg: dict, task_type: str) -> dict:
    narrow_params = {}

    # Tree depth
    depth_cfg = narrow_params_cfg.get("model", {}).get("depth")
    if "Model__depth" in best_params and depth_cfg and depth_cfg.get("include", False):
        offsets, low, high = get_default_int_params(
            depth_cfg,
            default_offsets=[1, 2],
            default_low=2,
            default_high=12
        )
        narrow_params["Model__depth"] = refine_int(
            best_params["Model__depth"],
            offsets=offsets,
            low=low,
            high=high
        )

    # Learning rate
    learning_rate_cfg = narrow_params_cfg.get("model", {}).get("learning_rate")
    if "Model__learning_rate" in best_params and learning_rate_cfg and learning_rate_cfg.get("include", False):
        factors, low, high, decimals = get_default_float_params(
            learning_rate_cfg,
            default_factors=[0.7, 0.85, 1.0, 1.15, 1.3],
            default_low=0.003,
            default_high=0.5,
            default_decimals=5
        )
        narrow_params["Model__learning_rate"] = refine_float_mult(
            best_params["Model__learning_rate"],
            factors=factors,
            low=low,
            high=high,
            decimals=decimals
        )
    
    # L2 regularization
    l2_leaf_reg_cfg = narrow_params_cfg.get("model", {}).get("l2_leaf_reg")
    if "Model__l2_leaf_reg" in best_params and l2_leaf_reg_cfg and l2_leaf_reg_cfg.get("include", False):
        factors, low, high, decimals = get_default_float_params(
            l2_leaf_reg_cfg,
            default_factors=[0.7, 0.85, 1.0, 1.15, 1.3],
            default_low=1.0,
            default_high=30.0,
            default_decimals=2
        )
        narrow_params["Model__l2_leaf_reg"] = refine_float_mult(
            best_params["Model__l2_leaf_reg"],
            factors=factors,
            low=low,
            high=high,
            decimals=decimals
        )

    # Bagging temperature
    bagging_temperature_cfg = narrow_params_cfg.get("ensemble", {}).get("bagging_temperature")
    if "Model__bagging_temperature" in best_params and bagging_temperature_cfg and bagging_temperature_cfg.get("include", False):
        factors, low, high, decimals = get_default_float_params(
            bagging_temperature_cfg,
            default_factors=[0.6, 0.8, 1.0, 1.2, 1.5],
            default_low=0.0,
            default_high=5.0,
            default_decimals=3
        )
        narrow_params["Model__bagging_temperature"] = refine_float_mult(
            best_params["Model__bagging_temperature"],
            factors=factors,
            low=low,
            high=high,
            decimals=decimals
        )

    # Min data in leaf
    min_data_in_leaf_cfg = narrow_params_cfg.get("model", {}).get("min_data_in_leaf")
    if "Model__min_data_in_leaf" in best_params and min_data_in_leaf_cfg and min_data_in_leaf_cfg.get("include", False):
        offsets, low, high = get_default_int_params(
            min_data_in_leaf_cfg,
            default_offsets=[2, 5, 10],
            default_low=1,
            default_high=50
        )
        narrow_params["Model__min_data_in_leaf"] = refine_int(
            best_params["Model__min_data_in_leaf"],
            offsets=offsets,
            low=low,
            high=high
        )

    # Random strength
    random_strength_cfg = narrow_params_cfg.get("model", {}).get("random_strength")
    if "Model__random_strength" in best_params and random_strength_cfg and random_strength_cfg.get("include", False):
        factors, low, high, decimals = get_default_float_params(
            random_strength_cfg,
            default_factors=[0.7, 0.85, 1.0, 1.15, 1.3],
            default_low=0.0,
            default_high=20.0,
            default_decimals=2
        )
        narrow_params["Model__random_strength"] = refine_float_mult(
            best_params["Model__random_strength"],
            factors=factors,
            low=low,
            high=high,
            decimals=decimals
        )

    # Freeze GPU-sensitive / discretization params
    border_count_cfg = narrow_params_cfg.get("model", {}).get("border_count")
    if "Model__border_count" in best_params and border_count_cfg and border_count_cfg.get("include", False):
        narrow_params["Model__border_count"] = refine_border_count(
            best_params["Model__border_count"]
        )

    colsample_bylevel_cfg = narrow_params_cfg.get("model", {}).get("colsample_bylevel")
    if "Model__colsample_bylevel" in best_params and colsample_bylevel_cfg and colsample_bylevel_cfg.get("include", False):
        factors, low, high, decimals = get_default_float_params(
            colsample_bylevel_cfg,
            default_factors=[0.9, 1.0, 1.1],
            default_low=0.5,
            default_high=1.0,
            default_decimals=2
        )
        narrow_params["Model__colsample_bylevel"] = [1.0] if task_type == "GPU" else refine_float_mult(
            best_params["Model__colsample_bylevel"],
            factors=factors,
            low=low,
            high=high,
            decimals=decimals
        )

    return narrow_params