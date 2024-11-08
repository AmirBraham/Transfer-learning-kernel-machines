# evaluate.py

import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score

def evaluate(preds, y_true):
    """
    Evaluates predictions against true labels.

    Args:
        preds (np.ndarray): Predicted labels, shape (n_samples, num_classes).
        y_true (np.ndarray): One-hot encoded true labels, shape (n_samples, num_classes).

    Returns:
        dict: Dictionary containing MSE and classification accuracy.
    """
    eval_metrics = {}
    # Mean Squared Error
    mse = mean_squared_error(y_true, preds)
    eval_metrics['mse'] = mse

    # Classification Accuracy
    y_true_labels = np.argmax(y_true, axis=1)
    y_pred_labels = np.argmax(preds, axis=1)
    acc = accuracy_score(y_true_labels, y_pred_labels)
    eval_metrics['accuracy'] = acc

    return eval_metrics
