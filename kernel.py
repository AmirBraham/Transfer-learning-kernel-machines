# kernel.py

from neural_tangents import stax
import jax  # Import JAX
from jax import jit  # Import jit from JAX

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
    # JIT compile the kernel for faster computation using JAX
    return kernel_fn