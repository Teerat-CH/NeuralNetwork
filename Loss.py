from abc import ABC, abstractmethod
import numpy as np

class Loss(ABC):
    
    @abstractmethod
    def loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def loss_derivative(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        pass

class MeanSquaredError(Loss):
    def loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return 0.5 * np.mean((y_pred - y_true) ** 2)

    def loss_derivative(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return y_pred - y_true