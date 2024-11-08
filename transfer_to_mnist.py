# transfer_to_mnist.py

import numpy as np
import torch
import dataset as d
import kernel as k
import krr
import os

def main():
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Create directory to save models if it doesn't exist
    os.makedirs('saved_models', exist_ok=True)

    # Load CIFAR-10 training data and trained weights
    train_loader, _ = d.get_cifar10_data(batch_size=1000)
    X_cifar_train, y_cifar_train = d.normalized_numpy_data(train_loader, num_classes=10)
    X_cifar_train = X_cifar_train[:1]
    y_cifar_train = y_cifar_train[:1]

    # Load trained KRR weights
    try:
        weights = np.load('saved_models/cifar10_krr_weights.npy')
        print(f"Loaded KRR weights shape: {weights.shape}")  # Should match (n_train, num_classes)
    except FileNotFoundError:
        print("Trained KRR weights not found. Please run 'train_cifar_model.py' first.")
        return

    # Load MNIST data in grayscale
    mnist_train_loader, mnist_test_loader = d.get_mnist_data(batch_size=1000)
    X_mnist_train, y_mnist_train = d.normalized_numpy_data(mnist_train_loader, num_classes=10)
    X_mnist_test, y_mnist_test = d.normalized_numpy_data(mnist_test_loader, num_classes=10)

    # Ensure data is grayscale: (batch_size, height, width, 1)
    X_mnist_train = X_mnist_train[..., np.newaxis] if X_mnist_train.ndim == 3 else X_mnist_train
    X_mnist_test = X_mnist_test[..., np.newaxis] if X_mnist_test.ndim == 3 else X_mnist_test

    # For computational feasibility, use subsets (adjust as needed)
    X_mnist_train = X_mnist_train[:1]
    y_mnist_train = y_mnist_train[:1]
    X_mnist_test = X_mnist_test[:1]
    y_mnist_test = y_mnist_test[:1]

    print(f"Shape of MNIST training data: {X_mnist_train.shape}")  # (1000, 28, 28, 1)
    print(f"Shape of MNIST test data: {X_mnist_test.shape}")      # (1000, 28, 28, 1)

    # Define the kernel function
    kernel_fn = k.conv_net()

    # Compute the kernel matrix between MNIST training data and CIFAR-10 training data
    print("Computing kernel matrix for MNIST training data...")
    K_mnist_train = kernel_fn(X_mnist_train, X_cifar_train, get='ntk')
    K_mnist_train = np.array(K_mnist_train, dtype=np.float32)
    print(f"MNIST Training Kernel matrix shape: {K_mnist_train.shape}")

    # Train a new KRR model on MNIST using the CIFAR-10 kernel
    print("Training KRR model on MNIST using CIFAR-10 kernel...")
    alpha = 1e-6  # Regularization parameter
    krr_model = krr.train_krr(K_mnist_train, y_mnist_train, alpha=alpha)
    print("MNIST KRR model training completed.")

    # Save the trained MNIST KRR model
    np.save('saved_models/mnist_krr_weights.npy', krr_model.coef_)
    print("Trained MNIST KRR weights saved to 'saved_models/mnist_krr_weights.npy'.")

    # Compute the kernel matrix between MNIST test data and CIFAR-10 training data
    print("Computing kernel matrix for MNIST test data...")
    K_mnist_test = kernel_fn(X_mnist_test, X_cifar_train, get='ntk')
    K_mnist_test = np.array(K_mnist_test, dtype=np.float32)
    print(f"MNIST Test Kernel matrix shape: {K_mnist_test.shape}")

    # Make predictions on MNIST test data
    print("Making predictions on MNIST test data...")
    preds = krr.predict_krr(krr_model, K_mnist_test)
    print("Predictions completed.")

    # Evaluate the transferred model
    print("Evaluating transferred model performance on MNIST test data...")
    metrics = krr.evaluate(preds, y_mnist_test)
    print(f"MNIST Test MSE: {metrics['mse']:.5f}")
    print(f"MNIST Test Accuracy: {metrics['accuracy'] * 100:.2f}%")

if __name__ == '__main__':
    main()
