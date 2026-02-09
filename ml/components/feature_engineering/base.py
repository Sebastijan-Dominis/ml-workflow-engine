import pandas as pd

from ..base import PipelineComponent, SklearnFeatureMixin


class FeatureOperator(PipelineComponent):
    """Produces one or more features."""
    output_features: list[str] = []

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Override in child class."""
        raise NotImplementedError()


class FeatureEngineer(SklearnFeatureMixin):
    def __init__(self, derived_schema: pd.DataFrame, operators: dict):
        self.derived_schema = derived_schema
        self.operators = operators

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for op_name in self.derived_schema["source_operator"].unique():
            operator = self.operators[op_name]
            X = operator.transform(X)

            # Validate outputs exist
            for f in operator.output_features:
                if f not in X.columns:
                    raise ValueError(f"{op_name} did not produce expected feature '{f}'")
        return X
