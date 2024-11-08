# svd.py

import numpy as np
import torch

def nystrom_kernel_svd(samples, kernel_fn, k):
    """
    Computes the SVD of the kernel matrix using the Nystrom method.

    Args:
        samples (np.ndarray): Data samples of shape (n_sample, H, W, C).
        kernel_fn (callable): Kernel function from neural_tangents.
        k (int): Number of top eigenvalues and eigenvectors to retain.

    Returns:
        eigvals (np.ndarray): Top k eigenvalues.
        eigvecs (np.ndarray): Corresponding eigenvectors.
    """
    # Ensure samples are NumPy arrays
    if isinstance(samples, torch.Tensor):
        samples = samples.cpu().numpy()

    # Compute the kernel matrix with 'k_train_train' component
    try:
        kmat = kernel_fn(samples, samples, get='ntk_train_train')
    except TypeError:
        # If 'get' parameter is not supported, call without it
        kmat = kernel_fn(samples, samples)

    # Check available keys and extract the correct one
    possible_keys = ['ntk_train_train', 'ntk']
    for key in possible_keys:
        if key in kmat:
            kmat = kmat[key]
            print(f"Using kernel key: {key}")
            break
    else:
        available_keys = list(kmat.keys())
        raise KeyError(f"None of the expected keys {possible_keys} found in kernel_fn output. Available keys: {available_keys}")

    print(f"Kernel matrix shape ({key}): {kmat.shape}")

    # Convert to NumPy array if not already
    if not isinstance(kmat, np.ndarray):
        kmat = np.array(kmat, dtype=np.float32)

    # Perform SVD using eigh since the kernel matrix is symmetric
    eigvals, eigvecs = np.linalg.eigh(kmat)

    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigvals)[::-1]
    eigvals = eigvals[sorted_indices]
    eigvecs = eigvecs[:, sorted_indices]

    # Select the top k eigenvalues and eigenvectors
    eigvals = np.real(eigvals[:k])
    eigvecs = np.real(eigvecs[:, :k])

    print(f"Top {k} eigenvalues: {eigvals}")
    print(f"Top {k} eigenvectors shape: {eigvecs.shape}")

    return eigvals, eigvecs