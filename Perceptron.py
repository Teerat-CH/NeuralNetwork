import numpy as np
from Activation import Activation

class Perceptron:
    def __init__(self, input_size: int, activation: Activation) -> None:
        self.input_size = input_size
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn(1)[0]
        self.output: float = None
        self.weighted_sum: float = None
        self.activation = activation

    def forward(self, input: np.ndarray) -> float:
        self.input = input
        self.weighted_sum = np.dot(self.weights, input) + self.bias
        self.output = self.activation.activate(self.weighted_sum)
        return self.output

    def backward(self, gradients: float, learning_rate: float) -> np.ndarray:
        grad_wrt_weighted_sum: float = gradients * self.activation.activate_derivative(self.weighted_sum)
        grad_wrt_weights: np.ndarray = grad_wrt_weighted_sum * self.input
        grad_wrt_input: np.ndarray = grad_wrt_weighted_sum * self.weights
        grad_wrt_bias: float = grad_wrt_weighted_sum
        
        self.weights -= learning_rate * grad_wrt_weights
        self.bias -= learning_rate * grad_wrt_bias
        
        return grad_wrt_input

    def __str__(self):
        pass