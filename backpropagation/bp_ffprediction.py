import numpy as np

# sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# feedforward_prediction
def bp_feedforward_prediction(x, W):
    z = [np.array(0) for i in range(len(W) + 1)]
    y = [np.array(0) for i in range(len(W) + 1)]
    y[0] = x
    for i in np.arange(1, len(W) + 1):
        z[i] = y[i - 1].dot(W[i - 1])
        y[i] = sigmoid(z[i])
    return y[len(W)]

x = np.array([0.75, 0.25])
W1 = np.array([[0.2, 0.4], [0.8, 0.5]])
W2 = np.array([[0.6], [0.2]])
x2 = np.array([0.58661758, 0.60467908])
W = [W1, W2]
y = bp_feedforward_prediction(x = x, W = W)
print(y)

