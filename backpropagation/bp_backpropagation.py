import numpy as np

import inspect

# sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# backpropagation
def bp_backpropagation(x, W, ytrue, lbd = 1):
    leng = len(W)
    z = [np.array([0]) for i in range(leng + 1)]
    y = [np.array([0]) for i in range(leng + 1)]
    y[0] = x
    for i in np.arange(1, leng + 1):
        z[i] = y[i - 1].dot(W[i - 1])
        y[i] = sigmoid(z[i])
    delta = [np.array([0]) for i in range(leng + 1)]
    Wnew = [np.array([[0]]) for i in range(leng)]
    delta[leng] = (ytrue - y[leng]) * y[i] * (1 - y[i])
    for i in np.arange(leng - 1, -1, -1):
        c = delta[i + 1].shape
        Wnew[i] = W[i]
        for k in np.arange(W[i].shape[0]):
            delta[i] = np.zeros(W[i].shape[0])
            for j in np.arange(W[i].shape[1]):
                Wnew[i][k, j] += delta[i + 1][j] * y[i][k]
                delta[i][k] += delta[i+1][j] * W[i][k, j] * y[i][k] * (1 - y[i][k])
    return Wnew

# feedforward_prediction
def bp_feedforward_prediction(x, W):
    z = [np.array([0]) for i in range(len(W) + 1)]
    y = [np.array([0]) for i in range(len(W) + 1)]
    y[0] = x
    for i in np.arange(1, len(W) + 1):
        z[i] = y[i - 1].dot(W[i - 1])
        y[i] = sigmoid(z[i])
    return y[len(W)]

# t = np.array([[1, 2], [3, 4]])
# print("t[0, 1] = ", t[0, 1])
# t = [np.array([1, 2]), np.array([3, 5])]
x = np.array([0.75, 0.25])
W1 = np.array([[0.4, 0.2], [0.1, 0.8]])
W2 = np.array([[0.3], [0.7]])
yt = 0.5

W = [W1, W2]
print("W = ", W)
y = bp_feedforward_prediction(x = x, W = W)
print("y = ", y)
W = bp_backpropagation(x = x, W = W, ytrue = yt, lbd = 1)
print("W = ", W)
y = bp_feedforward_prediction(x = x, W = W)
print("y = ", y)