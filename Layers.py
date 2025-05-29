from Perceptron import Perceptron
from Activation import Activation
import numpy as np 

class FeedForward:
    def __init__(self, input_size: int, num_perceptrons: int, activation: Activation):
        self.perceptrons = [Perceptron(input_size, activation) for _ in range(num_perceptrons)]

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        return np.array([p.forward(x) for p in self.perceptrons])

    def backward(self, gradient: np.ndarray, learning_rate: float):
        total_grad = np.zeros_like(self.perceptrons[0].input, dtype=np.float64)
        for i, p in enumerate(self.perceptrons):
            grad_i = p.backward(gradient[i], learning_rate)
            total_grad += grad_i
        return total_grad
    
class Convolutional:
    def __init__(self, input_shape, filter_size: int, num_filters: int):
        input_depth, input_height, input_width = input_shape
        self.input_shape = input_shape
        self.filter_size = filter_size
        self.num_filters = num_filters

        # array of filters, each filter is a 3D array (depth, height, width)
        self.filters = np.random.randn(num_filters, input_depth, filter_size, filter_size)
        # array of biases, one for each filter, output an array of matrices
        self.biases = np.random.randn(num_filters, input_height - filter_size + 1, input_width - filter_size + 1)

    def convolve(input: np.ndarray, filter: np.ndarray, stride: int = 1) -> np.ndarray:
        input_depth, input_height, input_width = input.shape
        filter_depth, filter_height, filter_width = filter.shape

        assert input_depth == filter_depth, "Input and filter depth must match"

        output_height = (input_height - filter_height) // stride + 1
        output_width = (input_width - filter_width) // stride + 1

        output = np.zeros((output_height, output_width))

        for i in range(output_height):
            for j in range(output_width):
                row = i * stride
                col = j * stride
                region = input[:, row:row+filter_height, col:col+filter_width]
                output[i, j] = np.sum(region * filter)

        return output

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        self.output = np.copy(self.bias)

        for i in range(self.num_filters): # for each filter
            for j in range(self.input_shape[0]): # for each input channel
                self.output[i] += self.convolve(x[j], self.filters[i, j]) # convolve the input matrix with the filter
        return self.output

    def backward(self, output_gradient):
        # Implement the backward pass for the convolutional layer
        pass
