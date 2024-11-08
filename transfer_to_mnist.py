# transfer_to_mnist.py

import numpy as np
import torch
import dataset as d
import kernel as k
import krr
import os
import matplotlib.pyplot as plt

def main():
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Create directory to save models and results if they don't exist
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # Load CIFAR-10 training data and trained weights
    train_loader, _ = d.get_cifar10_data(batch_size=1000)
    X_cifar_train, y_cifar_train = d.normalized_numpy_data(train_loader, num_classes=10)
    
    # Load MNIST data in grayscale
    mnist_train_loader, mnist_test_loader = d.get_mnist_data(batch_size=1000)
    X_mnist_train, y_mnist_train = d.normalized_numpy_data(mnist_train_loader, num_classes=10)
    X_mnist_test, y_mnist_test = d.normalized_numpy_data(mnist_test_loader, num_classes=10)

    # Ensure data is grayscale: (batch_size, height, width, 1)
    X_mnist_train = X_mnist_train[..., np.newaxis] if X_mnist_train.ndim == 3 else X_mnist_train
    X_mnist_test = X_mnist_test[..., np.newaxis] if X_mnist_test.ndim == 3 else X_mnist_test

    # Loop to increase both CIFAR-10 and MNIST sample sizes and track accuracy
    cifar_sample_sizes = [10,50,100,200,300,500,700]  # CIFAR-10 sample sizes
    mnist_sample_sizes = [10, 50, 100,200,300]  # MNIST sample sizes
    results = {}

    for cifar_samples in cifar_sample_sizes:
        results[cifar_samples] = []

        # Use a subset of the CIFAR-10 data
        X_cifar_train_subset = X_cifar_train[:cifar_samples]
        y_cifar_train_subset = y_cifar_train[:cifar_samples]

        # Load trained KRR weights
        try:
            weights = np.load(f'saved_models/cifar10_krr_weights_{cifar_samples}.npy')
            print(f"Loaded KRR weights for {cifar_samples} CIFAR samples.")
        except FileNotFoundError:
            print(f"Trained KRR weights for {cifar_samples} CIFAR samples not found. Please run 'train_cifar_model.py' first.")
            continue

        for mnist_samples in mnist_sample_sizes:
            print(f"Transferring with {cifar_samples} CIFAR samples and {mnist_samples} MNIST samples...")

            # Use a subset of the MNIST data
            X_mnist_train_subset = X_mnist_train[:mnist_samples]
            y_mnist_train_subset = y_mnist_train[:mnist_samples]
            X_mnist_test_subset = X_mnist_test[:10]  # Keep the test set small for speed
            y_mnist_test_subset = y_mnist_test[:10]

            # Define the kernel function
            kernel_fn = k.conv_net()

            # Compute the kernel matrix for MNIST training data and CIFAR-10 training data
            print("Computing kernel matrix for MNIST training data...")
            K_mnist_train = kernel_fn(X_mnist_train_subset, X_cifar_train_subset, get='ntk')
            K_mnist_train = np.array(K_mnist_train, dtype=np.float32)
            print(f"MNIST Training Kernel matrix shape: {K_mnist_train.shape}")

            # Train a new KRR model on MNIST using the CIFAR-10 kernel
            print("Training KRR model on MNIST using CIFAR-10 kernel...")
            alpha = 1e-6  # Regularization parameter
            krr_model = krr.train_krr(K_mnist_train, y_mnist_train_subset, alpha=alpha)
            print("MNIST KRR model training completed.")

            # Compute the kernel matrix for MNIST test data
            print("Computing kernel matrix for MNIST test data...")
            K_mnist_test = kernel_fn(X_mnist_test_subset, X_cifar_train_subset, get='ntk')
            K_mnist_test = np.array(K_mnist_test, dtype=np.float32)
            print(f"MNIST Test Kernel matrix shape: {K_mnist_test.shape}")

            # Make predictions on MNIST test data
            print("Making predictions on MNIST test data...")
            preds = krr.predict_krr(krr_model, K_mnist_test)
            print("Predictions completed.")

            # Evaluate the transferred model
            print("Evaluating transferred model performance on MNIST test data...")
            metrics = krr.evaluate(preds, y_mnist_test_subset)
            accuracy = metrics['accuracy'] * 100
            print(f"MNIST Test MSE: {metrics['mse']:.5f}")
            print(f"MNIST Test Accuracy: {accuracy:.2f}%")

            # Store results
            results[cifar_samples].append((mnist_samples, accuracy))

    # Plot and save the results
    for cifar_samples, accuracy_data in results.items():
        mnist_sizes, accuracies = zip(*accuracy_data)
        plt.figure()
        plt.plot(mnist_sizes, accuracies, marker='o')
        plt.title(f'Accuracy vs. MNIST Samples (CIFAR Samples: {cifar_samples})')
        plt.xlabel('Number of MNIST Samples')
        plt.ylabel('Accuracy (%)')
        plt.grid(True)
        plt.savefig(f'results/accuracy_cifar_{cifar_samples}.png')
        plt.close()

    print("All results saved as images in the 'results' directory.")

if __name__ == '__main__':
    main()
