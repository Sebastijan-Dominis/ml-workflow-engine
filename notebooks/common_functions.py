import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

def optimal_f1_search(pipeline, X, y_true):
    
    y_probs = pipeline.predict_proba(X)[:, 1]

    thresholds = np.linspace(0,1,101)
    f1_scores = []

    for t in thresholds:
        y_pred_thresh = (y_probs >= t).astype(int)
        f1_scores.append(f1_score(y_true, y_pred_thresh))

    best_idx = np.argmax(f1_scores)
    print("Best threshold:", thresholds[best_idx])
    print("Best F1:", f1_scores[best_idx])

    plt.plot(thresholds, f1_scores)
    plt.xlabel("Threshold")
    plt.ylabel("F1 Score")
    plt.title("F1 Score vs Decision Threshold")
    plt.show()

    return thresholds[best_idx]