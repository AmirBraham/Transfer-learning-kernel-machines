# train_cifar_model.py

import numpy as np
import torch
import dataset as d
import kernel as k
import krr
import os
import matplotlib.pyplot as plt

def to_grayscale(X):
    # Average over the color channels to convert to grayscale
    return X.mean(axis=-1, keepdims=True)

def main():
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Create directories to save models and results if they don't exist
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # Load CIFAR-10 data
    train_loader, test_loader = d.get_cifar10_data(batch_size=1000)
    X_train, y_train = d.normalized_numpy_data(train_loader, num_classes=10)
    X_test, y_test = d.normalized_numpy_data(test_loader, num_classes=10)

    # Convert to grayscale
    X_train = to_grayscale(X_train)
    X_test = to_grayscale(X_test)

    # List of subset sizes to test
    subset_sizes = [10,50,100,200,300,500,700]
    results = []

    for subset_size in subset_sizes:
        print(f"Training with {subset_size} samples...")

        # Use a subset of the CIFAR-10 data
        X_train_subset = X_train[:subset_size]
        y_train_subset = y_train[:subset_size]
        X_test_subset = X_test[:100]  # Keep the test set small for speed
        y_test_subset = y_test[:100]

        # Define the kernel function
        kernel_fn = k.conv_net()

        # Compute the kernel matrix for training data
        print("Computing training kernel matrix...")
        K_train = kernel_fn(X_train_subset, X_train_subset, 'ntk')
        K_train = np.array(K_train, dtype=np.float32)
        print(f"Training kernel matrix shape: {K_train.shape}")

        # Train Kernel Ridge Regression model
        print("Training Kernel Ridge Regression model...")
        alpha = 1e-6  # Regularization parameter
        model = krr.train_krr(K_train, y_train_subset, alpha=alpha)
        print("Model training completed.")

        # Save the trained model weights
        model_path = f'saved_models/cifar10_krr_weights_{subset_size}.npy'
        np.save(model_path, model.coef_)
        print(f"Trained weights saved to '{model_path}'.")

        # Compute the kernel matrix for test data
        print("Computing test kernel matrix...")
        K_test = kernel_fn(X_test_subset, X_train_subset, 'ntk')
        K_test = np.array(K_test, dtype=np.float32)
        print(f"Test kernel matrix shape: {K_test.shape}")

        # Make predictions on test data
        print("Making predictions on test data...")
        preds = krr.predict_krr(model, K_test)
        print("Predictions completed.")

        # Evaluate the model
        print("Evaluating model performance...")
        metrics = krr.evaluate(preds, y_test_subset)
        accuracy = metrics['accuracy'] * 100
        print(f"Test MSE: {metrics['mse']:.5f}")
        print(f"Test Accuracy: {accuracy:.2f}%")

        # Store the results
        results.append((subset_size, accuracy))

    # Plot and save the results
    subset_sizes, accuracies = zip(*results)
    plt.figure()
    plt.plot(subset_sizes, accuracies, marker='o')
    plt.title('Accuracy vs. CIFAR-10 Subset Size')
    plt.xlabel('Number of Training Samples')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.savefig('results/accuracy_vs_subset_size.png')
    plt.close()

    print("Accuracy plot saved as 'results/accuracy_vs_subset_size.png'.")

if __name__ == '__main__':
    main()
