import numpy as np
class CosineKernel:

    def __init__(self, data):
        self.data = data

    def cosine_kernel1(self, u):
        abs_u = np.abs(u)
        kernel_values = (np.pi / 4) * np.cos((np.pi / 2) * u)
        kernel_values[abs_u > 1] = 0
        return kernel_values

    def cosine_kernel2(self, u):
        abs_u = np.abs(u)
        kernel_values = 1 + np.cos(2 * np.pi * u)
        kernel_values[abs_u > 0.5] = 0
        return kernel_values

    def cosine_kernel_density_estimate(self, x, bandwidth):
        n = len(self.data)
        d = self.data.shape[1] if len(self.data.shape) > 1 else 1
        distances = np.linalg.norm(x - self.data, axis=1)
        kernel_values = self.cosine_kernel1(distances / bandwidth)
        density_estimate = np.sum(kernel_values) / (n * (bandwidth ** d))
        return density_estimate
