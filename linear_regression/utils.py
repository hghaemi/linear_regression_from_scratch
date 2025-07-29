import numpy as np


def generate_data(n_samples=100, noise=0.1, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    
    X = np.random.randn(n_samples, 1)
    true_weight = np.random.uniform(-5, 5)
    true_bias = np.random.uniform(-3, 3)
    y = true_weight * X.flatten() + true_bias + noise * np.random.randn(n_samples)
    return X, y


def generate_multivariate_data(n_samples=100, n_features=3,
                                noise=0.1, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    
    X = np.random.randn(n_samples, n_features)
    
    true_weights = np.random.uniform(-3, 3, n_features)
    true_bias = np.random.uniform(-2, 2)
    
    y = np.dot(X, true_weights) + true_bias + noise * np.random.randn(n_samples)
    return X, y, true_weights, true_bias