from Layer import Layer
from Activation import Activation
from Loss import Loss
import numpy as np
from typing import List

class Network:
    def __init__(self, layer_sizes: np.ndarray, loss: Loss, activations: List[Activation]):
        self.layers = []
        self.loss = loss
        if len(layer_sizes) -1 != len(activations):
            raise ValueError("Number of activation functions must match the number of layers.")
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i+1], activations[i]))

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray, learning_rate: float):
        loss_grad:np.ndarray = self.loss.loss_derivative(y_pred, y_true)
        for layer in reversed(self.layers):
            loss_grad = layer.backward(loss_grad, learning_rate)

    def train(self, x, y, learning_rate):
        y_pred = self.forward(x)
        self.backward(y_pred, y, learning_rate)
        return y_pred