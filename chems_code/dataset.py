import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader

def get_cifar10_data(batch_size=1000):
    """
    Load CIFAR-10 data and return data loaders.
    """
    transform = transforms.Compose([
        transforms.Grayscale(),  # Convert to grayscale
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def get_mnist_data(batch_size=100):
    """
    Load MNIST data and return data loaders.
    """
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize MNIST images to 32x32
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def normalized_numpy_data(loader, num_classes=10):
    """ 
    Normalize data and convert to numpy arrays.
    """
    data_list = []
    labels_list = []
    for data, labels in loader:
        data_list.append(data.view(data.size(0), -1).numpy())  # Flatten the images
        labels_list.append(labels.numpy())
    
    data = np.concatenate(data_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    return data, np.eye(num_classes)[labels]  # One-hot encode labels
