# krr.py

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import OneHotEncoder

def train_krr(K_train, y_train, alpha=1e-6):
    """
    Trains Kernel Ridge Regression model.

    Args:
        K_train (np.ndarray): Kernel matrix for training data, shape (n_train, n_train).
        y_train (np.ndarray): One-hot encoded training labels, shape (n_train, num_classes).
        alpha (float): Regularization parameter.

    Returns:
        Ridge: Trained Ridge regression model.
    """
    # Initialize Ridge regression model with precomputed kernel
    krr = Ridge(alpha=alpha, fit_intercept=False)
    krr.fit(K_train, y_train)
    return krr

def predict_krr(krr, K_test):
    """
    Makes predictions using the trained KRR model.

    Args:
        krr (Ridge): Trained Ridge regression model.
        K_test (np.ndarray): Kernel matrix for test data, shape (n_test, n_train).

    Returns:
        np.ndarray: Predictions, shape (n_test, num_classes).
    """
    preds = krr.predict(K_test)
    return preds

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
