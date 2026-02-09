from dataclasses import dataclass


@dataclass(frozen=True)
class ParamConstraints:
    min_value: float | int | None = None
    max_value: float | int | None = None
    allow_zero: bool = True
    allow_negative: bool = False


CATBOOST_PARAM_CONSTRAINTS = {
    "depth": ParamConstraints(
        min_value=1,
        max_value=16,
        allow_zero=False
    ),
    "learning_rate": ParamConstraints(
        min_value=1e-6,
        max_value=1.0,
        allow_zero=False
    ),
    "l2_leaf_reg": ParamConstraints(
        min_value=0.0,
        max_value=100.0,
        allow_zero=True
    ),
    "random_strength": ParamConstraints(
        min_value=0.0,
        max_value=100.0,
        allow_zero=True
    ),
    "min_data_in_leaf": ParamConstraints(
        min_value=1,
        max_value=10_000,
        allow_zero=False
    ),
    "bagging_temperature": ParamConstraints(
        min_value=0.0,
        max_value=10.0,
        allow_zero=True
    ),
    "border_count": ParamConstraints(
        min_value=1,
        max_value=254,
        allow_zero=False
    ),
    "colsample_bylevel": ParamConstraints(
        min_value=0.0,
        max_value=1.0,
        allow_zero=False
    ),
}
