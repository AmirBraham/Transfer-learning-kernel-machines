# train_cifar_model.py

import numpy as np
import torch
import dataset as d
import kernel as k
import krr
import os



def to_grayscale(X):
    # Average over the color channels to convert to grayscale
    return X.mean(axis=-1, keepdims=True)


def main():
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Create directory to save models if it doesn't exist
    os.makedirs('saved_models', exist_ok=True)

    # Load CIFAR-10 data
    train_loader, test_loader = d.get_cifar10_data(batch_size=1000)
    X_train, y_train = d.normalized_numpy_data(train_loader, num_classes=10)
    X_test, y_test = d.normalized_numpy_data(test_loader, num_classes=10)

    # For computational feasibility, use a subset (adjust as needed)
    X_train = X_train[:100]
    y_train = y_train[:100]
    X_test = X_test[:10]
    y_test = y_test[:10]

    # Convert to grayscale
    X_train = to_grayscale(X_train)
    X_test = to_grayscale(X_test)

    
    
    print(f"Shape of X_train: {X_train.shape}")  # Should be (5000, 32, 32, 3)
    print(f"Shape of y_train: {y_train.shape}")  # Should be (5000, 10)
    print(f"Shape of X_test: {X_test.shape}")    # Should be (1000, 32, 32, 3)
    print(f"Shape of y_test: {y_test.shape}")    # Should be (1000, 10) 

    # Define the kernel function
    kernel_fn = k.conv_net()
    
    # Compute the kernel matrix for training data
    print("Computing training kernel matrix...")
    K_train = kernel_fn(X_train, X_train, 'ntk')  # Use 'ntk' to specify the Neural Tangents Kernel
    print(K_train)
    #K_train = K_train['ntk']
    K_train = np.array(K_train, dtype=np.float32)
    print(f"Training kernel matrix shape: {K_train.shape}")

    # Train Kernel Ridge Regression model
    print("Training Kernel Ridge Regression model...")
    alpha = 1e-6  # Regularization parameter
    model = krr.train_krr(K_train, y_train, alpha=alpha)
    print("Model training completed.")

    # Save the trained model
    np.save('saved_models/cifar10_krr_weights.npy', model.coef_)
    print("Trained weights saved to 'saved_models/cifar10_krr_weights.npy'.")

    # Compute the kernel matrix for test data
    print("Computing test kernel matrix...")
    K_test = kernel_fn(X_test, X_train,'ntk')
    K_test = np.array(K_test, dtype=np.float32)
    print(f"Test kernel matrix shape: {K_test.shape}")

    # Make predictions on test data
    print("Making predictions on test data...")
    preds = krr.predict_krr(model, K_test)
    print("Predictions completed.")

    # Evaluate the model
    print("Evaluating model performance...")
    metrics = krr.evaluate(preds, y_test)
    print(f"Test MSE: {metrics['mse']:.5f}")
    print(f"Test Accuracy: {metrics['accuracy'] * 100:.2f}%")

if __name__ == '__main__':
    main()
