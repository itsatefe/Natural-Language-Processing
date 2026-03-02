import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self, learning_rate=0.001, num_iterations=5000, lambda_=0.5):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.lambda_ = lambda_
        self.theta = None
        self.cost_history = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def compute_cost(self, X, y, theta):
        m = len(y)
        h = self.sigmoid(X.dot(theta))
        epsilon = 1e-5
        reg_term = (self.lambda_ / (2 * m)) * np.sum(np.square(theta[1:]))
        cost = -1/m * (y.dot(np.log(h + epsilon)) + (1 - y).dot(np.log(1 - h + epsilon))) + reg_term
        return cost

    def fit(self, X, y, epsilon=1e-7):
        X_bias = np.c_[np.ones((X.shape[0], 1)), X].astype(np.float64)
        self.theta = np.zeros(X_bias.shape[1])
        for i in range(self.num_iterations):
            h = self.sigmoid(X_bias.dot(self.theta))
            gradient = (X_bias.T.dot(y - h) + self.lambda_ * np.r_[np.array([0.]), self.theta[1:]]) / len(y)
            self.theta += self.learning_rate * gradient
            cost = self.compute_cost(X_bias, y, self.theta)
            self.cost_history.append(cost)
            if i > 0 and abs(self.cost_history[i - 1] - self.cost_history[i]) < epsilon:
                print(f"Converged after {i} iterations.")
                break

    def plot_cost_history(self):
        plt.plot(self.cost_history)
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.title("Cost Function Over Iterations")
        plt.show()

    def predict(self, X):
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]
        y_pred_prob = self.sigmoid(X_bias.dot(self.theta))
        y_pred = (y_pred_prob >= 0.5).astype(int)
        return y_pred
