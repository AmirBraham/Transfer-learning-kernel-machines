import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Enhanced RBF Kernel function implemented in PyTorch
def rbf_kernel(X, Y, gamma=0.01):
    X_norm = (X ** 2).sum(1).view(-1, 1)
    Y_norm = (Y ** 2).sum(1).view(1, -1)
    K = torch.exp(-gamma * (X_norm + Y_norm - 2.0 * torch.mm(X, Y.T)))
    return K.to(device)

# Kernel Ridge Regression class using PyTorch
class KernelRidgeRegression:
    def __init__(self, alpha=1e-6, gamma=0.01):
        self.alpha = alpha
        self.gamma = gamma
        self.alpha_ = None
        self.X_train_ = None

    def fit(self, X, y):
        X, y = X.to(device), y.to(device)
        K = rbf_kernel(X, X, self.gamma)
        n_samples = K.size(0)
        I = torch.eye(n_samples, device=K.device)
        self.alpha_ = torch.linalg.solve(K + self.alpha * I, y)
        self.X_train_ = X

    def predict(self, X):
        X = X.to(device)
        K_test = rbf_kernel(X, self.X_train_, self.gamma)
        return K_test @ self.alpha_

    def save_model(self, path):
        torch.save({
            'alpha_': self.alpha_,
            'X_train_': self.X_train_,
            'alpha': self.alpha,
            'gamma': self.gamma
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.alpha_ = checkpoint['alpha_']
        self.X_train_ = checkpoint['X_train_']
        self.alpha = checkpoint['alpha']
        self.gamma = checkpoint['gamma']

# Load and preprocess CIFAR-10 data using PyTorch
def load_cifar10_data(batch_size=1000):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    X_train_list, y_train_list = [], []
    for data, labels in trainloader:
        X_train_list.append(data)
        y_train_list.append(labels)
    X_train = torch.cat(X_train_list, dim=0).view(X_train.size(0), -1)
    y_train = torch.eye(10)[torch.cat(y_train_list, dim=0)]

    X_test_list, y_test_list = [], []
    for data, labels in testloader:
        X_test_list.append(data)
        y_test_list.append(labels)
    X_test = torch.cat(X_test_list, dim=0).view(X_test.size(0), -1)
    y_test = torch.cat(y_test_list, dim=0)

    return X_train, y_train, X_test, y_test

# Main code
if __name__ == '__main__':
    # Load data
    X_train, y_train, X_test, y_test = load_cifar10_data()

    # Use a subset for faster training
    subset_size = 1000
    X_train_subset = X_train[:subset_size].to(device)
    y_train_subset = y_train[:subset_size].to(device)
    X_test_subset = X_test[:100].to(device)
    y_test_subset = y_test[:100].to(device)

    # Train the Kernel Ridge Regression model
    krr_model = KernelRidgeRegression(alpha=1e-6, gamma=0.01)
    print("Training KRR model...")
    krr_model.fit(X_train_subset, y_train_subset)

    # Save the trained model
    os.makedirs('saved_results', exist_ok=True)
    model_path = 'saved_results/krr_model.pth'
    krr_model.save_model(model_path)
    print(f"Model saved to {model_path}")

    # Make predictions
    print("Making predictions...")
    y_pred = krr_model.predict(X_test_subset)
    y_pred_classes = torch.argmax(y_pred, dim=1)

    # Evaluate model
    accuracy = (y_pred_classes == y_test_subset).float().mean().item()
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Plot a few test predictions
    plt.figure(figsize=(10, 5))
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(X_test_subset[i].cpu().view(32, 32), cmap='gray')
        plt.title(f"Pred: {y_pred_classes[i].item()}")
        plt.axis('off')
    plt.show()
