import numpy as np
from scipy.stats import multivariate_normal
from sklearn.utils import check_X_y, check_array


class GaussianMixtureModel:
    def __init__(self, n_components, max_iter=100, tol=1e-4, reg_covar=1e-6):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        
    def initialize_parameters(self, data):
        
        kmeans = KMeans(n_clusters=self.n_components,random_state=42)
        labels = kmeans.fit_predict(data)
        self.means = kmeans.centroids

        self.n_samples, self.n_features = data.shape
        self.covariances = np.tile(np.eye(self.n_features), (self.n_components, 1, 1))
        self.weights = np.ones(self.n_components) / self.n_components
        # covariance is full
        self.num_params = self.n_components * (self.n_features + self.n_features * (self.n_features + 1) // 2) - 1


    def expectation_step(self, data):
        self.posteriors = np.zeros((data.shape[0], self.n_components))
        for i in range(self.n_components):
            covar = self.covariances[i] + self.reg_covar * np.eye(data.shape[1])
            likelihood = multivariate_normal.pdf(data, mean=self.means[i], cov=covar)
            self.posteriors[:, i] = self.weights[i] * likelihood.reshape(-1)

        self.posteriors /= self.posteriors.sum(axis=1, keepdims=True)

    def maximization_step(self, data):
        self.weights = self.posteriors.mean(axis=0)
        self.means = np.dot(self.posteriors.T, data) / self.posteriors.sum(axis=0)[:, np.newaxis]
        for i in range(self.n_components):
            diff = data - self.means[i]
            covar = np.dot(self.posteriors[:, i] * diff.T, diff) / self.posteriors[:, i].sum()
            self.covariances[i] = covar + self.reg_covar * np.eye(self.n_features)

    def fit(self, data):
        self.initialize_parameters(data)
        self.prev_log_likelihood = -np.inf
        for iteration in range(self.max_iter):
            self.expectation_step(data)
            self.maximization_step(data)
            log_likelihood = self.log_likelihood(data)
            if np.abs(self.prev_log_likelihood - log_likelihood) < self.tol:
                break
            self.prev_log_likelihood = log_likelihood
          

    def log_likelihood(self, data):
        likelihoods = np.zeros((self.n_samples, self.n_components))
        for i in range(self.n_components):
            covar = self.covariances[i] + self.reg_covar * np.eye(self.n_features)
            likelihoods[:, i] = multivariate_normal.pdf(data, mean=self.means[i], cov=covar)
        return np.log(np.dot(likelihoods, self.weights)).sum()

    def predict(self, data):
        self.expectation_step(data)
        return np.argmax(self.posteriors, axis=1)
    
    # The lower the better
    def bic(self, X):
        penalty_term = self.num_params * np.log(self.n_samples)
        return -2 * self.prev_log_likelihood * self.n_samples +  penalty_term

     # The lower the better
    def aic(self, X):
        return -2 * self.prev_log_likelihood * self.n_samples + 2 * self.num_params


class KMeans:
    def __init__(self, n_clusters=2, max_iter=100, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.centroids = None
        self.inertia_ = None 

    def _initialize_centroids(self, X):
        np.random.seed(self.random_state)
        random_idx = np.random.permutation(X.shape[0])
        centroids = X[random_idx[:self.n_clusters]]
        return centroids

    def _compute_distances(self, X, centroids):
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        return distances

    def _assign_clusters(self, distances):
        return np.argmin(distances, axis=0)

    def _update_centroids(self, X, labels):
        centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])
        return centroids

    def fit(self, X):
        self.centroids = self._initialize_centroids(X)
        self.inertia_ = 0

        for _ in range(self.max_iter):
            old_centroids = self.centroids
            distances = self._compute_distances(X, old_centroids)
            self.labels_ = self._assign_clusters(distances)
            self.centroids = self._update_centroids(X, self.labels_)
            self.inertia_ = self._compute_inertia(X, self.labels_)

            if np.all(old_centroids == self.centroids):
                break

        return self

    def _compute_inertia(self, X, labels):
        inertia = 0
        for i in range(self.n_clusters):
            cluster_points = X[labels == i]
            inertia += ((cluster_points - self.centroids[i]) ** 2).sum()
        return inertia

    def predict(self, X):
        distances = self._compute_distances(X, self.centroids)
        return self._assign_clusters(distances)

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_

    

class SoftmaxClassifier:
    def __init__(self, learning_rate=0.01, num_iterations=1000, regularization_strength=0.1):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.regularization_strength = regularization_strength

    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _compute_cost(self, X, y):
        m = len(y)
        scores = X.dot(self.theta_)
        probabilities = self._softmax(scores)
        log_probabilities = -np.log(probabilities[range(m), y])
        cost = np.sum(log_probabilities) / m
        reg_cost = (self.regularization_strength / (2 * m)) * np.sum(self.theta_[1:] ** 2)
        return cost + reg_cost

    def _compute_gradient(self, X, y):
        m = len(y)
        scores = X.dot(self.theta_)
        probabilities = self._softmax(scores)
        one_hot_labels = np.zeros_like(probabilities)
        one_hot_labels[np.arange(len(y)), y] = 1
        gradient = -X.T.dot(one_hot_labels - probabilities) / m
        gradient[1:] += (self.regularization_strength / m) * self.theta_[1:]
        return gradient

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        num_classes = len(self.classes_)
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]
        self.theta_ = np.zeros((X_bias.shape[1], num_classes))

        for i in range(self.num_iterations):
            gradient = self._compute_gradient(X_bias, y)
            self.theta_ -= self.learning_rate * gradient
            # Optional: Implement convergence check and early stopping here

    def predict(self, X):
        X = check_array(X)
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]
        scores = X_bias.dot(self.theta_)
        probabilities = self._softmax(scores)
        return np.argmax(probabilities, axis=1)

    def predict_prob(self, X):
        X = check_array(X)
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]
        scores = X_bias.dot(self.theta_)
        return self._softmax(scores)

class OvA:
    def __init__(self, binary_classifier):
        self.binary_classifiers = []
        self.binary_classifier = binary_classifier
        self.cost_history = []

    def train(self, X, y):
        unique_classes = np.unique(y)
        for class_label in unique_classes:
            binary_labels = (y == class_label).astype(int)
            classifier = self.binary_classifier()
            classifier.fit(X, binary_labels)
            self.binary_classifiers.append((class_label, classifier))
            self.cost_history.append((classifier.cost_history,f'{class_label} vs All'))

    def predict(self, X):
        predictions = []
        for _, classifier in self.binary_classifiers:
            prediction = classifier.predict(X)
            predictions.append(prediction)
        predicted_classes = np.argmax(predictions, axis=0)
        return predicted_classes

    
class OvO:
    def __init__(self, binary_classifier):
        self.binary_classifiers = []
        self.binary_classifier = binary_classifier
        self.unique_classes = None
        self.cost_history = []

    def train(self, X, y):
        self.unique_classes = np.unique(y)
        num_classes = len(self.unique_classes)

        for i in range(num_classes):
            for j in range(i + 1, num_classes):
                class_indices = np.logical_or(y == self.unique_classes[i], y == self.unique_classes[j])
                X_pair, y_pair = X[class_indices], y[class_indices]
                binary_labels = (y_pair == self.unique_classes[i]).astype(int)
                classifier = self.binary_classifier()
                classifier.fit(X_pair, binary_labels)
                self.binary_classifiers.append((self.unique_classes[i], self.unique_classes[j], classifier))
                self.cost_history.append((classifier.cost_history,f"class {i} vs {j}"))

    def predict(self, X):
        num_instances = X.shape[0]
        num_classifiers = len(self.binary_classifiers)
        votes = np.zeros((num_classifiers, num_instances))
        for k, (_, _, classifier) in enumerate(self.binary_classifiers):
            binary_prediction = classifier.predict(X)
            votes[k, :] = binary_prediction

        class_votes = np.zeros((len(self.unique_classes), num_instances))
        for k, (class1, class2, _) in enumerate(self.binary_classifiers):
            class_votes[class1, :] += (votes[k, :] == 1)
            class_votes[class2, :] += (votes[k, :] == 0)
        predicted_classes = np.argmax(class_votes, axis=0)
        return predicted_classes 
    
    def predict_probability(self, X):
        num_instances = X.shape[0]
        num_classifiers = len(self.binary_classifiers)
        class_probabilities = np.zeros((len(self.unique_classes), num_instances))
        for k, (class1, class2, classifier) in enumerate(self.binary_classifiers):
            binary_probabilities = classifier.predict_probability(X)[:, 1]
            class_probabilities[class1, :] += binary_probabilities
            class_probabilities[class2, :] += 1 - binary_probabilities
        class_probabilities /= num_classifiers
        return class_probabilities







