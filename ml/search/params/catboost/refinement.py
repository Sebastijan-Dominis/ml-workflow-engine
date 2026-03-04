"""Narrow-search parameter refinement helpers for CatBoost searches."""

from ml.search.params.refiners import refine_border_count, refine_float_mult, refine_int
from ml.search.params.utils import get_default_float_params, get_default_int_params


def prepare_narrow_params(best_params: dict, narrow_params_cfg, task_type: str) -> dict:
    """Build narrow search distributions around broad-search best parameters.

    Args:
        best_params: Best-parameter mapping returned by broad search.
        narrow_params_cfg: Narrow-search configuration object with refinement rules.
        task_type: Compute task type (for example ``CPU`` or ``GPU``).

    Returns:
        Refined parameter distributions for narrow search.

    Notes:
        GPU-specific constraints are applied for selected parameters (for
        example ``colsample_bylevel``), and only params marked ``include`` in
        narrow config are expanded.

    Examples:
        >>> params = prepare_narrow_params(best_params, narrow_cfg, task_type="GPU")

    Side Effects:
        None.
    """

    narrow_params = {}

    # Tree depth
    depth_cfg = narrow_params_cfg.model.depth if narrow_params_cfg.model else None
    if "Model__depth" in best_params and depth_cfg and depth_cfg.include:
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
    learning_rate_cfg = narrow_params_cfg.model.learning_rate if narrow_params_cfg.model else None
    if "Model__learning_rate" in best_params and learning_rate_cfg and learning_rate_cfg.include:
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
    l2_leaf_reg_cfg = narrow_params_cfg.model.l2_leaf_reg if narrow_params_cfg.model else None
    if "Model__l2_leaf_reg" in best_params and l2_leaf_reg_cfg and l2_leaf_reg_cfg.include:
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
    bagging_temperature_cfg = narrow_params_cfg.ensemble.bagging_temperature if narrow_params_cfg.ensemble else None
    if "Model__bagging_temperature" in best_params and bagging_temperature_cfg and bagging_temperature_cfg.include:
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
    min_data_in_leaf_cfg = narrow_params_cfg.model.min_data_in_leaf if narrow_params_cfg.model else None
    if "Model__min_data_in_leaf" in best_params and min_data_in_leaf_cfg and min_data_in_leaf_cfg.include:
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
    random_strength_cfg = narrow_params_cfg.model.random_strength if narrow_params_cfg.model else None
    if "Model__random_strength" in best_params and random_strength_cfg and random_strength_cfg.include:
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
    border_count_cfg = narrow_params_cfg.model.border_count if narrow_params_cfg.model else None
    if "Model__border_count" in best_params and border_count_cfg and border_count_cfg.include:
        narrow_params["Model__border_count"] = refine_border_count(
            best_params["Model__border_count"]
        )

    colsample_bylevel_cfg = narrow_params_cfg.model.colsample_bylevel if narrow_params_cfg.model else None
    if "Model__colsample_bylevel" in best_params and colsample_bylevel_cfg and colsample_bylevel_cfg.include:
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
