from Network import Network
from Sigmoid import Sigmoid
import numpy as np

X = np.array([
    [0, 0, 1],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
])
Y = np.array([
    [0],
    [1],
    [1],
    [0]
])

net = Network([3, 4, 1], Sigmoid(), random_seed=14)

for epoch in range(100000):
    loss = 0
    for x, y in zip(X, Y):
        pred = net.train(x, y, learning_rate=0.1)
        loss += 0.5 * np.sum((pred - y) ** 2)
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
