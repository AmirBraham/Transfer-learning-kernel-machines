# dataset.py

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

def get_cifar10_data(batch_size=1000, num_workers=2):
    """
    Loads CIFAR-10 dataset.

    Args:
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses for data loading.

    Returns:
        DataLoader: Training DataLoader.
        DataLoader: Test DataLoader.
    """
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    training_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader

def get_mnist_data(batch_size=1000, num_workers=2):
    """
    Loads MNIST dataset.

    Args:
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses for data loading.

    Returns:
        DataLoader: Training DataLoader.
        DataLoader: Test DataLoader.
    """
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels
        transforms.ToTensor(),
    ])

    training_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader

def normalized_numpy_data(loader, num_classes):
    """
    Converts data from DataLoader to normalized NumPy arrays with one-hot encoding for labels.

    Args:
        loader (DataLoader): PyTorch DataLoader.
        num_classes (int): Number of classes for one-hot encoding.

    Returns:
        X (np.ndarray): Normalized image data of shape (N, H, W, C).
        Y (np.ndarray): One-hot encoded labels of shape (N, num_classes).
    """
    X, Y = [], []
    for inputs, labels in loader:
        # Convert images to NumPy arrays and transpose to (N, H, W, C)
        x = inputs.numpy().transpose(0, 2, 3, 1)  # From (N, C, H, W) to (N, H, W, C)
        x /= 255.0  # Normalize pixel values to [0, 1]
        y = np.zeros((len(labels), num_classes))
        y[np.arange(len(labels)), labels.numpy()] = 1.
        X.append(x)
        Y.append(y)
    X = np.concatenate(X, axis=0).astype('float32')
    Y = np.concatenate(Y, axis=0).astype('float32')
    return X, Y
