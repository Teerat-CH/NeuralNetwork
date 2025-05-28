from Perceptron import Perceptron
from Activation import Activation
import numpy as np

class Layer:
    def __init__(self, input_size, num_perceptrons, activation: Activation, random_seed=None):
        self.perceptrons = [Perceptron(input_size, activation, random_seed) for _ in range(num_perceptrons)]

    def forward(self, x):
        self.input = x
        return np.array([p.forward(x) for p in self.perceptrons])

    def backward(self, gradient: np.ndarray, learning_rate: float):
        total_grad = np.zeros_like(self.perceptrons[0].input, dtype=np.float64)
        for i, p in enumerate(self.perceptrons):
            grad_i = p.backward(gradient[i], learning_rate)
            total_grad += grad_i
        return total_grad
