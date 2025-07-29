"""
Linear Regression from Scratch
A simple implementation of linear regression using gradient descent.
"""

from .models import LinearRegression, LinearRegressionWithScaling
from .utils import generate_data, generate_multivariate_data

__version__ = "0.1.0"
__author__ = "M. Hossein Ghaemi"
__email__ = "h.ghaemi.2003@gmail.co"

__all__ = [
    "LinearRegression",
    "LinearRegressionWithScaling", 
    "generate_data",
    "generate_multivariate_data"
]
