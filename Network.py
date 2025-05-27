from Layer import Layer
import numpy as np

class Network:
    def __init__(self, layer_sizes, random_seed=None):
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i+1], random_seed))

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, y_pred, y_true, learning_rate):
        loss_grad = y_pred - y_true  # ∂L/∂a (for MSE loss)
        for layer in reversed(self.layers):
            loss_grad = layer.backward(loss_grad, learning_rate)

    def train(self, x, y, learning_rate):
        y_pred = self.forward(x)
        self.backward(y_pred, y, learning_rate)
        return y_pred


if __name__ == "__main__":
    # XOR inputs and labels
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    Y = np.array([
        [0],
        [1],
        [1],
        [0]
    ])

    # Define the network: 2 → 2 → 1
    net = Network([2, 16, 1],  random_seed=14)

    # Train it
    for epoch in range(100000):
        loss = 0
        for x, y in zip(X, Y):
            pred = net.train(x, y, learning_rate=0.1)
            loss += 0.5 * np.sum((pred - y) ** 2)
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
