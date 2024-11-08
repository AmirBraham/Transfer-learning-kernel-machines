import numpy as np

def conv_net():
    """
    Kernel function to compute the kernel matrix (e.g., RBF kernel or NTK).
    """
    def kernel_fn(X, Y, get='ntk'):
    # Ensure input matrices are aligned in shape
        return np.exp(-0.01 * (np.sum(X**2, axis=1, keepdims=True) +
                           np.sum(Y**2, axis=1) -
                           2 * np.dot(X, Y.T)))
    return kernel_fn
import numpy as np

def projection(kernel_fn, X_source, X_target):
    """
    Project the source data (e.g., CIFAR-10) into the feature space of the target data (e.g., MNIST).
    """
    print("Projecting source data to target feature space...")
    K_source_target = kernel_fn(X_source, X_target, get='ntk')
    return np.array(K_source_target, dtype=np.float32)

    