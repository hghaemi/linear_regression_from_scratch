import numpy as np 

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.bias = None
        self.weights = None
        self.learning_rate = learning_rate
        self.n_iterations  = n_iterations
        self.cost_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent
        for i in range(self.n_iterations ):
            y_pred = np.dot(X, self.weights) + self.bias

            # Compute cost function
            cost = (1/(2*n_samples)) * np.sum((y_pred - y) ** 2)
            self.cost_history.append(cost)

            # calculate derivate using vectors
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            # update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def get_cost_history(self):
        return self.cost_history


class LinearRegressionWithScaling(LinearRegression):
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        super().__init__(learning_rate, n_iterations)
        self.X_mean = None
        self.X_std = None
        self.y_mean = None
        self.y_std = None
    
    def fit(self, X, y):
        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0)
        self.y_mean = np.mean(y)
        self.y_std = np.std(y)
        
        X_scaled = (X - self.X_mean) / self.X_std
        y_scaled = (y - self.y_mean) / self.y_std
        
        super().fit(X_scaled, y_scaled)
    
    def predict(self, X):
        X_scaled = (X - self.X_mean) / self.X_std
        y_scaled_pred = super().predict(X_scaled)
        return y_scaled_pred * self.y_std + self.y_mean