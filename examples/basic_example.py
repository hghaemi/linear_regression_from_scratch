import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from linear_regression import LinearRegression, generate_data
import matplotlib.pyplot as plt


def main():
    print("Linear Regression - Basic Example")
    print("=" * 40)
    
    X, y = generate_data(n_samples=100, noise=0.2, random_seed=42)
    
    model = LinearRegression(learning_rate=0.01, n_iterations=1000)
    model.fit(X, y)
    
    y_pred = model.predict(X)
    
    mse = ((y_pred - y) ** 2).mean()
    r2 = 1 - ((y - y_pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    
    print(f"Learned weight: {model.weights[0]:.4f}")
    print(f"Learned bias: {model.bias:.4f}")
    print(f"Final cost: {model.cost_history[-1]:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X, y, alpha=0.6, label='Data points')
    plt.plot(X, y_pred, color='red', linewidth=2, label='Fitted line')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Linear Regression Fit')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(model.cost_history)
    plt.xlabel('Iterations')
    plt.ylabel('Cost (MSE)')
    plt.title('Training Progress')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
