import numpy as np

def train_krr(K, y, alpha):
    I = np.eye(K.shape[0])  # Identity matrix with the same shape as K
    alpha_ = np.linalg.solve(K + alpha * I, y)  # Solve (K + alpha * I) * alpha_ = y
    return alpha_


def predict_krr(alpha_, K_test):
    """
    Predict using the trained KRR model.
    """
    return np.dot(K_test, alpha_)

def evaluate(predictions, y_true):
    """
    Evaluate the performance of the model by calculating MSE and accuracy.
    """
    mse = np.mean((predictions - y_true) ** 2)
    accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(y_true, axis=1))
    return {'mse': mse, 'accuracy': accuracy}
        
def translate_predictions(predictions, mean, std):
    """
    Translate the predicted values to the target space.
    We can apply a translation that adjusts CIFAR predictions to MNIST space by normalizing them.
    """
    print("Translating predictions...")
    return (predictions - mean) / std
