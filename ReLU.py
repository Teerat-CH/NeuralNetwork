from Activation import Activation

class ReLU(Activation):
    def activate(self, value: float) -> float:
        return max(0, value)

    def activate_derivative(self, value: float) -> float:
        return 1.0 if value > 0 else 0.0