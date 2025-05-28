from Activation import Activation
import numpy as np

class Sigmoid(Activation):
    def activate(self, value: float) -> float:
        return 1 / (1 + np.exp(-value))

    def activate_derivative(self, value: float) -> float:
        return value * (1 - value)