from sklearn.base import BaseEstimator, TransformerMixin


class PipelineComponent(BaseEstimator, TransformerMixin):
    """Base class for all pipeline components."""
    def fit(self, X, y=None):
        return self
    
class SklearnFeatureMixin:
    """Mixin for transformers that must expose n_features_in_."""

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self
