from abc import ABC, abstractmethod

class Activation(ABC):
    
    @abstractmethod
    def activate(self, value: float) -> float:
        pass

    @abstractmethod
    def activate_derivative(self, value: float) -> float:
        pass