# projection.py

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import OneHotEncoder

def laplace_kernel(X1, X2, bandwidth=1.0):
    """
    Computes the Laplace kernel between two sets of vectors.

    Args:
        X1 (np.ndarray): First set of vectors, shape (n1, d).
        X2 (np.ndarray): Second set of vectors, shape (n2, d).
        bandwidth (float): Bandwidth parameter for the Laplace kernel.

    Returns:
        np.ndarray: Kernel matrix of shape (n1, n2).
    """
    # Compute the L1 distance between each pair of samples
    dists = np.sum(np.abs(X1[:, np.newaxis] - X2), axis=2)
    K = np.exp(-dists / bandwidth)
    return K

def learn_projection_kernel_method(source_preds_train, y_target_train, bandwidth=1.0, reg=1e-6):
    """
    Learns the projection mapping from source predictions to target labels using Kernel Ridge Regression.

    Args:
        source_preds_train (np.ndarray): Source model predictions on training data, shape (n_train, num_classes).
        y_target_train (np.ndarray): One-hot encoded target training labels, shape (n_train, num_classes).
        bandwidth (float): Bandwidth parameter for the Laplace kernel.
        reg (float): Regularization parameter for Kernel Ridge Regression.

    Returns:
        KernelRidge: Trained Kernel Ridge Regression model.
    """
    # Initialize Kernel Ridge Regression with precomputed kernel
    krr = KernelRidge(kernel='precomputed', alpha=reg)

    # Compute the kernel matrix between training predictions
    K_train = laplace_kernel(source_preds_train, source_preds_train, bandwidth=bandwidth)

    # Fit the KRR model
    krr.fit(K_train, y_target_train)

    return krr

def predict_with_projection_kernel_method(krr, source_preds_test, source_preds_train, bandwidth=1.0):
    """
    Makes predictions on test data using the learned projection mapping.

    Args:
        krr (KernelRidge): Trained Kernel Ridge Regression model.
        source_preds_test (np.ndarray): Source model predictions on test data, shape (n_test, num_classes).
        source_preds_train (np.ndarray): Source model predictions on training data, shape (n_train, num_classes).
        bandwidth (float): Bandwidth parameter for the Laplace kernel.

    Returns:
        np.ndarray: Predicted labels for the test data, shape (n_test, num_classes).
    """
    # Compute the kernel matrix between test predictions and training predictions
    K_test = laplace_kernel(source_preds_test, source_preds_train, bandwidth=bandwidth)

    # Predict using the KRR model
    preds = krr.predict(K_test)
    return preds
