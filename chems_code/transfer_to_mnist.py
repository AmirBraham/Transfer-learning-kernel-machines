import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import dataset as d
import kernel as k
import krr

def main():
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Create directories to save models and results if they don't exist
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # Load CIFAR-10 training data and trained weights
    train_loader, _ = d.get_cifar10_data(batch_size=1000)
    X_cifar_train, y_cifar_train = d.normalized_numpy_data(train_loader, num_classes=10)
    
    # Load MNIST data in grayscale
    mnist_train_loader, mnist_test_loader = d.get_mnist_data(batch_size=100)
    X_mnist_train, y_mnist_train = d.normalized_numpy_data(mnist_train_loader, num_classes=10)
    X_mnist_test, y_mnist_test = d.normalized_numpy_data(mnist_test_loader, num_classes=10)

    # Loop to increase both CIFAR-10 and MNIST sample sizes and track accuracy
    cifar_sample_sizes = [ 100, 200, 300]
    mnist_sample_sizes = [100]  # Use a fixed number of MNIST samples for testing
    results = {}

    for cifar_samples in cifar_sample_sizes:
        results[cifar_samples] = []

        # Use a subset of the CIFAR-10 data
        X_cifar_train_subset = X_cifar_train[:cifar_samples]
        y_cifar_train_subset = y_cifar_train[:cifar_samples]

        # Load trained KRR weights
        try:
            weights = np.load(f'saved_results/krr_model.pth', allow_pickle=True)
            print(f"Loaded KRR weights for {cifar_samples} CIFAR samples.")
        except FileNotFoundError:
            print(f"Trained KRR weights for {cifar_samples} CIFAR samples not found. Please run 'train_cifar_model.py' first.")
            continue

        for mnist_samples in mnist_sample_sizes:
            print(f"Transferring with {cifar_samples} CIFAR samples and {mnist_samples} MNIST samples...")

            # Use a subset of the MNIST data
            X_mnist_train_subset = X_mnist_train[:mnist_samples]
            y_mnist_train_subset = y_mnist_train[:mnist_samples]
            X_mnist_test_subset = X_mnist_test[:10]  # Smaller test set for demo
            y_mnist_test_subset = y_mnist_test[:10]

            # Define the kernel function
            kernel_fn = k.conv_net()

            # Projection: Compute kernel matrix between CIFAR-10 and MNIST data
            print("Projecting CIFAR-10 data to MNIST feature space...")
            K_mnist_train = k.projection(kernel_fn, X_cifar_train_subset, X_mnist_train_subset)
            print(f"MNIST Training Kernel matrix shape: {K_mnist_train.shape}")

            # Train a new KRR model on MNIST using the CIFAR-10 kernel
            print("Training KRR model on MNIST using CIFAR-10 kernel...")
            alpha = 1e-6  # Regularization parameter
            krr_model = krr.train_krr(K_mnist_train, y_mnist_train_subset, alpha=alpha)
            print("MNIST KRR model training completed.")

            # Compute the kernel matrix for MNIST test data
            print("Computing kernel matrix for MNIST test data...")
            K_mnist_test = k.projection(kernel_fn, X_cifar_train_subset, X_mnist_test_subset)
            print(f"MNIST Test Kernel matrix shape: {K_mnist_test.shape}")

            # Make predictions on MNIST test data
            print("Making predictions on MNIST test data...")
            preds = krr.predict_krr(krr_model, K_mnist_test)
            print("Predictions completed.")

            # Translate predictions into MNIST space
            preds_translated = krr.translate_predictions(preds, mean=np.mean(preds), std=np.std(preds))

            # Evaluate the transferred model
            print("Evaluating transferred model performance on MNIST test data...")
            metrics = krr.evaluate(preds_translated, y_mnist_test_subset)
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
