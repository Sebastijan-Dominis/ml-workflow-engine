from ml.runners.evaluation.evaluators.classification.classification import \
    EvaluateClassification
from ml.runners.evaluation.evaluators.regression.regression import \
    EvaluateRegression

EVALUATORS = {
    "classification": EvaluateClassification,
    "regression": EvaluateRegression
}