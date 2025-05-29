from Network import Network
from Activation import Sigmoid, ReLU
from Loss import MeanSquaredError
import numpy as np
np.random.seed(14)

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

sigmoid = Sigmoid()

net = Network([2, 2, 1], MeanSquaredError(), [sigmoid, sigmoid])

for epoch in range(1000):
    loss = 0
    for x, y in zip(X, Y):
        pred = net.train(x, y, learning_rate=0.5)
        loss += 0.5 * np.sum((pred - y) ** 2)
    if epoch % 1 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
