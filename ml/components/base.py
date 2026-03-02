"""Base classes and mixins for scikit-learn-compatible pipeline components."""

from sklearn.base import BaseEstimator, TransformerMixin


class PipelineComponent(BaseEstimator, TransformerMixin):
    """Base class for all pipeline components."""

    def fit(self, X, y=None):
        """Return ``self`` to satisfy scikit-learn estimator interface.

        Args:
            X: Input feature matrix.
            y: Optional target vector.

        Returns:
            PipelineComponent: Fitted component instance.
        """
        return self
    
class SklearnFeatureMixin:
    """Mixin for transformers that must expose n_features_in_."""

    def fit(self, X, y=None):
        """Store input feature count on fit for downstream compatibility.

        Args:
            X: Input feature matrix.
            y: Optional target vector.

        Returns:
            SklearnFeatureMixin: Fitted mixin instance.
        """
        self.n_features_in_ = X.shape[1]
        return self
