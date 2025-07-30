# Linear Regression from Scratch

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> A comprehensive, educational implementation of linear regression using gradient descent, built entirely from scratch without scikit-learn dependencies.

## 🎯 Overview

This project demonstrates a deep understanding of machine learning fundamentals by implementing linear regression with gradient descent from first principles. Perfect for educational purposes, interviews, or as a foundation for more complex algorithms.

### 🔬 What Makes This Special?

- **Pure Mathematics**: No black-box libraries - every calculation is explicit and understandable
- **Production-Ready Code**: Professional package structure with proper testing and documentation
- **Educational Focus**: Clear explanations of mathematical concepts and implementation details
- **Scalable Design**: Works with any number of features, from simple to high-dimensional data
- **Feature Engineering**: Includes automatic feature scaling for improved convergence

## ✨ Features

### Core Functionality
- ✅ **Pure NumPy Implementation** - No scikit-learn dependencies
- ✅ **Gradient Descent Optimization** - Manual implementation of the core ML algorithm
- ✅ **Multi-feature Support** - Handles datasets with any number of features
- ✅ **Feature Scaling** - Built-in standardization for better convergence
- ✅ **Cost Tracking** - Monitor training progress and convergence
- ✅ **Comprehensive Testing** - Full test suite ensuring reliability

### Mathematical Foundation
- **Cost Function**: Mean Squared Error (MSE)
- **Optimization**: Batch Gradient Descent
- **Feature Scaling**: Z-score normalization
- **Parameter Updates**: Vectorized operations for efficiency

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/hghaemi/linear_regression_from_scratch.git
cd linear_regression_from_scratch

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Basic Usage

```python
from linear_regression import LinearRegression, generate_data
import matplotlib.pyplot as plt

# Generate synthetic data
X, y = generate_data(n_samples=100, noise=0.2, random_seed=42)

# Create and train model
model = LinearRegression(learning_rate=0.01, n_iterations=1000)
model.fit(X, y)

# Make predictions
predictions = model.predict(X)

# Visualize results
plt.scatter(X, y, alpha=0.6, label='Data')
plt.plot(X, predictions, 'r-', label='Fitted Line')
plt.legend()
plt.show()

print(f"Learned weight: {model.weights[0]:.4f}")
print(f"Learned bias: {model.bias:.4f}")
print(f"Final cost: {model.cost_history[-1]:.4f}")
```

### Advanced Usage - Multiple Features

```python
from linear_regression import LinearRegression, generate_multivariate_data

# Generate data with 5 features
X, y, true_weights, true_bias = generate_multivariate_data(
    n_samples=200, 
    n_features=5, 
    noise=0.1, 
    random_seed=42
)

# Train model
model = LinearRegression(learning_rate=0.01, n_iterations=1500)
model.fit(X, y)

# Evaluate performance
predictions = model.predict(X)
mse = ((predictions - y) ** 2).mean()
r2 = 1 - ((y - predictions) ** 2).sum() / ((y - y.mean()) ** 2).sum()

print(f"MSE: {mse:.4f}")
print(f"R² Score: {r2:.4f}")
```

### Feature Scaling for Better Performance

```python
from linear_regression import LinearRegressionWithScaling

# For datasets with features at different scales
model_scaled = LinearRegressionWithScaling(learning_rate=0.1, n_iterations=1000)
model_scaled.fit(X, y)

# Automatically handles scaling internally
predictions = model_scaled.predict(X)
```

## 📁 Project Structure

```
linear-regression-from-scratch/
├── linear_regression/           # Core package
│   ├── __init__.py             # Package initialization
│   ├── models.py               # LinearRegression classes
│   └── utils.py                # Data generation utilities
├── examples/                   # Usage examples
│   ├── basic_example.py        # Simple regression demo
│   ├── multivariate_example.py # Multiple features demo
├── tests/                      # Test suite
│   ├── __init__.py
│   └── test_models.py          # Comprehensive tests
├── setup.py                    # Package installation
├── requirements.txt            # Dependencies
├── requirements-dev.txt        
├── README.md                   # This file
├── .gitignore                  # Git ignore rules
└── LICENSE                     # MIT License
```

## 🧮 Mathematical Foundation

### Linear Regression Model
```
y = X·w + b
```
Where:
- `y`: target values
- `X`: feature matrix
- `w`: weight vector
- `b`: bias term

### Cost Function (Mean Squared Error)
```
J(w,b) = (1/2m) Σ(ŷᵢ - yᵢ)²
```

### Gradient Descent Updates
```
w := w - α · (1/m) · Xᵀ·(ŷ - y)
b := b - α · (1/m) · Σ(ŷ - y)
```

Where `α` is the learning rate and `m` is the number of samples.

## 📊 Performance Analysis

### Learning Rate Impact
Different learning rates affect convergence:
- **Too low**: Slow convergence
- **Too high**: Oscillation or divergence
- **Optimal**: Smooth, fast convergence

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Using pytest
pytest tests/

# Or run tests manually
python tests/test_models.py
```

The test suite covers:
- Model initialization
- Single and multi-feature training
- Prediction accuracy
- Cost function convergence
- Data generation utilities
- Edge cases and error handling

## 🎓 Educational Value

This implementation is ideal for:
- **Students**: Understanding ML algorithms from first principles
- **Interviews**: Demonstrating deep algorithmic knowledge
- **Teaching**: Clear, well-documented code for instruction
- **Research**: Foundation for custom regression variants

### Key Learning Outcomes
- Gradient descent optimization
- Vectorized operations in NumPy
- Feature scaling importance
- Cost function design
- Professional Python packaging

## 🔧 API Reference

### LinearRegression Class

```python
class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000)
    def fit(self, X, y)                    # Train the model
    def predict(self, X)                   # Make predictions
    def get_cost_history(self)             # Get training history
```

### LinearRegressionWithScaling Class

```python
class LinearRegressionWithScaling(LinearRegression):
    # Inherits all methods from LinearRegression
    # Automatically scales features during training
```

### Utility Functions

```python
generate_data(n_samples=100, noise=0.1, random_seed=None)
generate_multivariate_data(n_samples=100, n_features=3, noise=0.1, random_seed=None)
```

## 🎯 Use Cases

- **Educational Projects**: Learn ML fundamentals
- **Prototyping**: Quick regression experiments
- **Baseline Models**: Compare against complex algorithms
- **Feature Engineering**: Test preprocessing pipelines
- **Algorithm Development**: Foundation for custom methods

## 🚧 Future Enhancements

- [ ] Regularization (Ridge, Lasso)
- [ ] Polynomial feature generation
- [ ] Cross-validation utilities
- [ ] More optimization algorithms (Adam, RMSprop)
- [ ] GPU acceleration with CuPy
- [ ] Sparse matrix support

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**M. Hossein Ghaemi**
- Email: h.ghaemi.2003@gmail.com
- GitHub: [@hghaemi](https://github.com/hghaemi)

## 🙏 Acknowledgments

- Inspired by Andrew Ng's Machine Learning Course
- Built with NumPy and Matplotlib
- Follows scikit-learn API conventions

---

*This project demonstrates a deep understanding of machine learning fundamentals and professional software development practices. Perfect for academic applications and technical interviews.*