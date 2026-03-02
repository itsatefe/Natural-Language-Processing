#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from sklearn.preprocessing import LabelEncoder

class FishersLDA:
    def __init__(self):
        self.mean = None
        self.std = None
        self.classes = None
        self.Sw = None
        self.Sb = None
        self.components = None

    def fit(self, X, y):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        X_standardized = self.data_whitening(X, self.mean, self.std)
        self.classes = np.unique(y)
        self.Sw = self.within_scatter(X_standardized, y)
        self.Sb = self.between_scatter(X_standardized, y)
        separability_matrix = self.calculate_separability_matrix(self.Sw, self.Sb)
        eigenvalues, eigenvectors = np.linalg.eig(separability_matrix)
        indices = np.argsort(eigenvalues)[::-1]
        # Take only the top n_classes - 1 eigenvectors
        num_components = min(len(self.classes) - 1, X.shape[1])
        self.components = eigenvectors[:, indices[:num_components]]

    def transform(self, X):
        X_standardized = self.data_whitening(X, self.mean, self.std)
        return np.dot(X_standardized, self.components)

    def data_whitening(self, X, mean, std):
        X = X - mean
        X = X / (std + 1e-10)
        return X

    def within_scatter(self, X, y):
        Sw = np.zeros((X.shape[1], X.shape[1]))
        for i in self.classes:
            Xi = X[y == i]
            Si = np.cov(Xi, rowvar=False) * (len(Xi) - 1)
            Sw += Si
        return Sw + np.eye(Sw.shape[0]) * 1e-10  # Regularization to prevent singular matrix

    def between_scatter(self, X, y):
        mean_overall = np.mean(X, axis=0)
        Sb = np.zeros((X.shape[1], X.shape[1]))
        for i in self.classes:
            Xi = X[y == i]
            mean_class_i = Xi.mean(axis=0)
            Ni = len(Xi)
            mean_diff = mean_class_i - mean_overall
            Sb += Ni * np.outer(mean_diff, mean_diff)
        return Sb

    def calculate_separability_matrix(self, Sw, Sb):
        Sw_inv = np.linalg.inv(Sw)
        return Sw_inv.dot(Sb)

