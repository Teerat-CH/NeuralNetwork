from abc import ABC, abstractmethod
import numpy as np

class Activation(ABC):
    
    @abstractmethod
    def activate(self, value: float) -> float:
        pass

    @abstractmethod
    def activate_derivative(self, value: float) -> float:
        pass

class Sigmoid(Activation):
    def activate(self, value: float) -> float:
        return 1 / (1 + np.exp(-value))

    def activate_derivative(self, value: float) -> float:
        a = self.activate(value)
        return a * (1 - a)

class ReLU(Activation):
    def __init__(self, alpha: float = 0):
        self.alpha = alpha

    def activate(self, value: float) -> float:
        return value if value > 0 else self.alpha * value

    def activate_derivative(self, value: float) -> float:
        return 1.0 if value > 0 else self.alpha