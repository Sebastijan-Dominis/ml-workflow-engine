"""Base abstractions and orchestration for feature engineering operators."""

import logging

import pandas as pd

from ml.exceptions import DataError

from ..base import PipelineComponent, SklearnFeatureMixin

logger = logging.getLogger(__name__)

class FeatureOperator(PipelineComponent):
    """Produces one or more features."""
    output_features: list[str] = []

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform input data by producing configured derived features.

        Args:
            X: Input dataframe for operator transformation.

        Returns:
            pd.DataFrame: Transformed dataframe including operator outputs.
        """

        raise NotImplementedError()


class FeatureEngineer(SklearnFeatureMixin):
    """Apply configured feature operators in schema-defined execution order."""

    def __init__(self, derived_schema: pd.DataFrame, operators: dict):
        """Initialize feature engineer with derived schema and operator map.

        Args:
            derived_schema: Derived schema describing operator execution order.
            operators: Mapping of operator name to instantiated operator.

        Returns:
            None: Initializes feature-engineering orchestrator state.
        """

        self.derived_schema = derived_schema
        self.operators = operators

    def fit(self, X, y=None):
        """Return ``self`` for scikit-learn compatibility.

        Args:
            X: Feature input ignored in this no-op fit.
            y: Optional target input ignored in this no-op fit.

        Returns:
            FeatureEngineer: Fitted estimator instance.
        """

        return self

    def transform(self, X):
        """Run all configured operators and validate expected outputs exist.

        Args:
            X: Input dataframe to transform.

        Returns:
            pd.DataFrame: Transformed dataframe after applying all operators.
        """

        X = X.copy()
        for op_name in self.derived_schema["source_operator"].unique():
            operator = self.operators[op_name]
            X = operator.transform(X)

            # Validate outputs exist
            for f in operator.output_features:
                if f not in X.columns:
                    msg = f"Operator '{op_name}' did not produce expected feature '{f}'"
                    logger.error(msg)
                    raise DataError(msg)
        return X
