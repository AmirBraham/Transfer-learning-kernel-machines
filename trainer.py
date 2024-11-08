# trainer.py

import numpy as np
import torch
import neural_tangents as nt
from neural_tangents import stax
import eigenpro

def nonlinearity(act_name='relu'):
    """
    Returns the activation function based on the given name.

    Args:
        act_name (str): Name of the activation function ('relu' or 'erf').

    Returns:
        callable: Activation function from neural_tangents.stax.
    """
    if act_name == 'relu':
        return stax.Relu()
    elif act_name == 'erf':
        return stax.Erf()
    else:
        raise ValueError("Unsupported activation function")

def conv_net(c=1.0, act_name='relu'):
    """
    Defines a convolutional network architecture and returns its kernel function.

    Args:
        c (float): Standard deviation for the final Dense layer weights.
        act_name (str): Activation function name.

    Returns:
        callable: Kernel function from neural_tangents.
    """
    _, _, kernel_fn = stax.serial(
        stax.Conv(out_chan=32, filter_shape=(3, 3), strides=(1, 1), padding='SAME'),
        nonlinearity(act_name),
        stax.Conv(out_chan=64, filter_shape=(3, 3), strides=(1, 1), padding='SAME'),
        nonlinearity(act_name),
        stax.Flatten(),
        stax.Dense(128),
        nonlinearity(act_name),
        stax.Dense(10, W_std=c)  # Assuming 10 classes
    )
    return kernel_fn

def train_kernel_model(X_train, y_train, num_epochs=10, batch_size=500):
    """
    Trains the kernel model using EigenPro on the provided training data.

    Args:
        X_train (np.ndarray): Training samples of shape (N, H, W, C).
        y_train (np.ndarray): One-hot encoded training labels of shape (N, num_classes).
        num_epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.

    Returns:
        np.ndarray: Trained weights of shape (n_centers, num_classes).
    """
    # Define the kernel function
    kernel_fn = conv_net()
    kernel_fn_batched = nt.batch(kernel_fn, device_count=-1, batch_size=50)

    # Initialize EigenPro model
    y_dim = y_train.shape[1]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = eigenpro.FKR_EigenPro(kernel_fn_batched, X_train, y_dim, device=device, name='cifar10_kernel_weights', compute_eval=True)

    # Prepare validation data (use a subset of training data for simplicity)
    x_val = X_train[:1000]
    y_val = y_train[:1000]

    # Train the model
    epochs = list(range(1, num_epochs + 1))
    model.fit(X_train, y_train, x_val, y_val, epochs=epochs, mem_gb=8)

    # Load the trained weights
    weights = np.load('saved_models/' + model.model_name + '.npy')
    return weights

def get_preds(X_train, X_test, weights):
    """
    Computes predictions using the trained kernel model.

    Args:
        X_train (np.ndarray): Training samples.
        X_test (np.ndarray): Test samples.
        weights (np.ndarray): Trained weights.

    Returns:
        np.ndarray: Predictions for the test samples.
    """
    kernel_fn = conv_net()
    kernel_fn_batched = nt.batch(kernel_fn, device_count=-1, batch_size=50)

    # Compute kernel matrix between test data and training data with 'ntk_train_train' component
    try:
        K_test_train = kernel_fn_batched(X_test, X_train, get='ntk_train_train')
    except TypeError:
        # If 'get' parameter is not supported, call without it
        K_test_train = kernel_fn_batched(X_test, X_train)

    # Check available keys and extract the correct one
    possible_keys = ['ntk_train_train', 'ntk']
    for key in possible_keys:
        if key in K_test_train:
            K_test_train = K_test_train[key]
            print(f"Using kernel key: {key} for K_test_train")
            break
    else:
        available_keys = list(K_test_train.keys())
        print(f"Available keys: {available_keys}")
        raise KeyError(f"None of the expected keys {possible_keys} found in kernel_fn output.")

    K_test_train = np.array(K_test_train, dtype=np.float32)

    # Get predictions
    preds = K_test_train @ weights
    return preds

def get_source_predictions_on_target(X_source_train, X_target, weights):
    """
    Computes source model predictions on target data.

    Args:
        X_source_train (np.ndarray): Source training samples.
        X_target (np.ndarray): Target samples.
        weights (np.ndarray): Trained weights.

    Returns:
        np.ndarray: Source model predictions on target data.
    """
    kernel_fn = conv_net()
    kernel_fn_batched = nt.batch(kernel_fn, device_count=-1, batch_size=50)

    # Compute kernel matrix between target data and source training data with 'ntk_train_train' component
    try:
        K_target_source = kernel_fn_batched(X_target, X_source_train, get='ntk_train_train')
    except TypeError:
        # If 'get' parameter is not supported, call without it
        K_target_source = kernel_fn_batched(X_target, X_source_train)

    # Check available keys and extract the correct one
    possible_keys = ['ntk_train_train', 'ntk']
    for key in possible_keys:
        if key in K_target_source:
            K_target_source = K_target_source[key]
            print(f"Using kernel key: {key} for K_target_source")
            break
    else:
        available_keys = list(K_target_source.keys())
        print(f"Available keys: {available_keys}")
        raise KeyError(f"None of the expected keys {possible_keys} found in kernel_fn output.")

    K_target_source = np.array(K_target_source, dtype=np.float32)

    # Get source model predictions on target data
    source_preds = K_target_source @ weights
    return source_preds
