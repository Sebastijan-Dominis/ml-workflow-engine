"""Registry mapping task types to evaluation runner implementations."""

from ml.runners.evaluation.evaluators.classification.classification import ClassificationEvaluator
from ml.runners.evaluation.evaluators.regression.regression import EvaluateRegression

EVALUATORS = {
    "classification": ClassificationEvaluator,
    "regression": EvaluateRegression
}
