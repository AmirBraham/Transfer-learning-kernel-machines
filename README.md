```markdown
# Transfer Learning with Kernel Methods

This repository contains code and resources for our project on transfer learning using kernel methods, inspired by the work of Radhakrishnan et al. (2023). The project explores the integration of kernel-based methods into transfer learning frameworks, specifically focusing on projection and translation techniques to adapt models across different tasks, such as transferring knowledge from CIFAR-10 to MNIST.

## Project Overview

Transfer learning is a vital machine learning technique that leverages knowledge from one task to improve performance on a related task, particularly when the target task has limited data. While deep neural networks are widely used for this purpose, they often act like "black boxes" and require substantial computational resources. In contrast, kernel methods offer simplicity and strong theoretical foundations but have been challenging to adapt for transfer learning across tasks with different label spaces or distributions.

This project introduces a framework for transfer learning with kernel methods, utilizing two key operations:
- **Projection**: Applying the trained source model to generate features for the target task.
- **Translation**: Adjusting the source model with a correction term to fit the target task.

## File Structure

The repository is organized as follows:

.
├── dataset.py                   # Data loading and preprocessing for CIFAR-10 and MNIST
├── evaluate.py                  # Functions for evaluating model performance (MSE, accuracy)
├── kernel.py                    # Definition of the convolutional kernel-based architecture
├── krr.py                       # Implementation of Kernel Ridge Regression (KRR)
├── projection.py                # Code for transferring predictions between datasets
├── svd.py                       # Approximate SVD computation using the Nystrom method
├── train_cifar_model.py         # Script to train the model on CIFAR-10
├── trainer.py                   # Utility functions for model training and data transfer
├── transfer_to_mnist.py         # Script to transfer the CIFAR-10 model to MNIST
├── utils.py                     # Helper functions for data normalization and preprocessing
├── README.md                    # Project documentation
```

## How to Run the Project

### Prerequisites

Ensure you have the following packages installed in your Python environment:
- `numpy`
- `scikit-learn`
- `neural-tangents`
- `matplotlib` (for visualization, if needed)

You can install the required packages using:
```bash
pip install numpy scikit-learn neural-tangents matplotlib
```

Installing Jax is a bit tricky so try to find a version that works depending on your version of CUDA

### Running the Project

1. **Data Preparation**: The `dataset.py` script handles data loading and preprocessing. Ensure that you have access to the CIFAR-10 and MNIST datasets. The script will transform the datasets into normalized NumPy arrays for use in training and evaluation.

2. **Training on CIFAR-10**:
   ```bash
   python train_cifar_model.py
   ```
   This script trains a kernel-based model on the CIFAR-10 dataset, computes kernel matrices, and saves the trained model weights.

3. **Transfer Learning to MNIST**:
   ```bash
   python transfer_to_mnist.py
   ```
   This script loads the saved weights from the CIFAR-10 model, applies transfer learning to the MNIST dataset, and evaluates performance.

4. **Evaluation**: Use `evaluate.py` to compute metrics like mean squared error (MSE) and classification accuracy on the test set.

### Explanation of Key Components

- **Projection**: The `projection.py` module handles the transfer of predictions from the source dataset (CIFAR-10) to the target dataset (MNIST) using Kernel Ridge Regression.
- **Kernel Methods**: The `kernel.py` script defines the convolutional network architecture, leveraging `neural-tangents` to compute kernel functions efficiently.

## References

Please refer to the paper by Radhakrishnan et al. (2023) for a detailed theoretical background and methodology. The reference is included in `sample.bib` for use in academic reports.

---

We hope this project inspires further exploration of transfer learning with kernel methods. Contributions and feedback are welcome!
```

