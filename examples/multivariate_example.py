import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from linear_regression import LinearRegression, generate_multivariate_data
import numpy as np
import matplotlib.pyplot as plt

def main():
    print("Linear Regression - Multivariate Example")
    print("=" * 45)
    
    feature_counts = [2, 5, 10]
    
    for n_features in feature_counts:
        print(f"\nTesting with {n_features} features:")
        print("-" * 30)
        
        X, y, true_weights, true_bias = generate_multivariate_data(
            n_samples=200, 
            n_features=n_features, 
            noise=0.1, 
            random_seed=42
        )
        
        model = LinearRegression(learning_rate=0.01, n_iterations=1500)
        model.fit(X, y)
        
        y_pred = model.predict(X)
        
        mse = ((y_pred - y) ** 2).mean()
        r2 = 1 - ((y - y_pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        
        weight_errors = np.abs(true_weights - model.weights)
        bias_error = abs(true_bias - model.bias)
        
        print(f"Final cost: {model.cost_history[-1]:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"R2 Score: {r2:.4f}")
        print(f"Average weight error: {np.mean(weight_errors):.4f}")
        print(f"Bias error: {bias_error:.4f}")
        
        n_show = min(3, n_features)
        print(f"True weights (first {n_show}): {true_weights[:n_show]}")
        print(f"Learned weights (first {n_show}): {model.weights[:n_show]}")
    
    plt.figure(figsize=(8, 5))
    plt.plot(model.cost_history)
    plt.xlabel('Iterations')
    plt.ylabel('Cost (MSE)')
    plt.title(f'Training Progress ({n_features} features)')
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    main()
