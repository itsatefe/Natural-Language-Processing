import numpy as np
from CosineKernel import CosineKernel

class KernelDensityClassifier:
    def __init__(self, bandwidth=0.2):
        self.bandwidth = bandwidth
        self.label_arrays = {}
        self.cosine_kernels = {}

    def fit(self, X_train, y_train):
        unique_labels = np.unique(np.argmax(y_train, axis=1))
        label_indices = {}

        for label in unique_labels:
            label_indices[label] = np.where(np.argmax(y_train, axis=1) == label)[0]

        for label, indices in label_indices.items():
            self.label_arrays[label] = X_train[indices]
            self.cosine_kernels[label] = CosineKernel(self.label_arrays[label])

    def predict(self, X_test):
        y_pred_density = []

        for i in range(len(X_test)):
            densities = []
            for label in self.cosine_kernels:
                density = self.cosine_kernels[label].cosine_kernel_density_estimate(X_test[i], self.bandwidth)
                densities.append(density)
            y_pred_density.append(densities)

        predicted_labels = np.argmax(y_pred_density, axis=1)
        return predicted_labels