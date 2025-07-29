import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from linear_regression import LinearRegression, generate_data, generate_multivariate_data

class TestLinearRegression:
    
    def test_initialization(self):
        model = LinearRegression(learning_rate=0.01, n_iterations=1000)
        assert model.learning_rate == 0.01
        assert model.n_iterations == 1000
        assert model.weights is None
        assert model.bias is None
        assert len(model.cost_history) == 0
    
    def test_fit_single_feature(self):
        X, y = generate_data(n_samples=50, noise=0.1, random_seed=42)
        model = LinearRegression(learning_rate=0.01, n_iterations=100)
        
        model.fit(X, y)
        
        assert model.weights is not None
        assert model.bias is not None
        assert len(model.weights) == 1
        assert len(model.cost_history) == 100
    
    def test_fit_multiple_features(self):
        X, y, _, _ = generate_multivariate_data(n_samples=50, n_features=3, random_seed=42)
        model = LinearRegression(learning_rate=0.01, n_iterations=100)
        
        model.fit(X, y)
        
        assert len(model.weights) == 3
        assert len(model.cost_history) == 100
    
    def test_predict(self):
        X, y = generate_data(n_samples=50, noise=0.01, random_seed=42)
        model = LinearRegression(learning_rate=0.01, n_iterations=500)
        
        model.fit(X, y)
        predictions = model.predict(X)
        
        assert len(predictions) == len(y)
        assert isinstance(predictions, np.ndarray)
    
    def test_cost_decreases(self):
        X, y = generate_data(n_samples=100, noise=0.1, random_seed=42)
        model = LinearRegression(learning_rate=0.01, n_iterations=500)
        
        model.fit(X, y)
        
        initial_cost = model.cost_history[0]
        final_cost = model.cost_history[-1]
        assert final_cost < initial_cost
    
    def test_perfect_data(self):
        np.random.seed(42)
        X = np.random.randn(100, 1)
        true_weight = 2.5
        true_bias = 1.0
        y = true_weight * X.flatten() + true_bias
        
        model = LinearRegression(learning_rate=0.01, n_iterations=1000)
        model.fit(X, y)
        
        assert abs(model.weights[0] - true_weight) < 0.01
        assert abs(model.bias - true_bias) < 0.01
    
    def test_get_cost_history(self):
        X, y = generate_data(n_samples=50, random_seed=42)
        model = LinearRegression(learning_rate=0.01, n_iterations=100)
        
        model.fit(X, y)
        cost_history = model.get_cost_history()
        
        assert len(cost_history) == 100
        assert cost_history == model.cost_history

def test_generate_data():
    X, y = generate_data(n_samples=100, noise=0.1, random_seed=42)
    
    assert X.shape == (100, 1)
    assert y.shape == (100,)
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)

def test_generate_multivariate_data():
    X, y, true_weights, true_bias = generate_multivariate_data(
        n_samples=100, n_features=5, random_seed=42
    )
    
    assert X.shape == (100, 5)
    assert y.shape == (100,)
    assert len(true_weights) == 5
    assert isinstance(true_bias, (int, float, np.number))

if __name__ == "__main__":
    test_class = TestLinearRegression()
    test_methods = [method for method in dir(test_class) if method.startswith('test_')]
    
    print("Running tests...")
    for method_name in test_methods:
        try:
            method = getattr(test_class, method_name)
            method()
            print(f"successful: {method_name}")
        except Exception as e:
            print(f"failed: {method_name}: {e}")
    
    try:
        test_generate_data()
        print("successful: test_generate_data")
    except Exception as e:
        print(f"failed: test_generate_data: {e}")
    
    try:
        test_generate_multivariate_data()
        print("successful: test_generate_multivariate_data")
    except Exception as e:
        print(f"failed: test_generate_multivariate_data: {e}")

    print("Tests completed!")