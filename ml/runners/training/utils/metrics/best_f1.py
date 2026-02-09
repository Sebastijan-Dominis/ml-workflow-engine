import logging

import numpy as np
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)

def get_best_f1_thresh(pipeline, X, y_true):
    y_probs = pipeline.predict_proba(X)[:, 1]

    thresholds = np.linspace(0,1,101)
    f1_scores = []

    for t in thresholds:
        y_pred_thresh = (y_probs >= t).astype(int)
        f1_scores.append(f1_score(y_true, y_pred_thresh))

    best_idx = np.argmax(f1_scores)
    return float(thresholds[best_idx]), float(f1_scores[best_idx])