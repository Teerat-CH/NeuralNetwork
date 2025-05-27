import numpy as np

class Perceptron:
    def __init__(self, input_size: int, random_seed: int=None) -> None:
        if random_seed is not None:
            np.random.seed(random_seed)
        self.input_size = input_size
        self.weights = np.random.randn(input_size)
        self.input: np.ndarray = None
        self.output: float = None
        self.weighted_sum: float = None

    def sigmoid(self, x: float) -> float:
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x: float) -> float:
        return x * (1 - x)

    def forward(self, input: np.ndarray) -> float:
        self.input = input
        self.weighted_sum = np.dot(self.weights, input)
        self.output = self.sigmoid(self.weighted_sum)
        return self.output

    def backward(self, gradients: float, learning_rate: float) -> np.ndarray:
        dz: float = gradients * self.sigmoid_derivative(self.output)
        dw: np.ndarray = dz * self.input
        self.weights -= learning_rate * dw
        return self.weights * dz

    def __str__(self):
        pass