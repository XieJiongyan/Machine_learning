import numpy as np

# feedforward_prediction
def bp_feedforward_prediction(x, W):
    y = [np.array(0) for i in range(len(W) + 1)]
    y[0] = x
    for i in np.arange(1, len(W) + 1):
        y[i] = y[i - 1].dot(W[i - 1])
    return y[len(W)]

x = np.array([0.75, 0.25])
W1 = np.array([[0.2, 0.4], [0.8, 0.5]])
W2 = np.array([[0.6], [0.2]])
W = [W1, W2]
y = bp_feedforward_prediction(x = x, W = W)
print(y)