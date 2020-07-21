import numpy as np


def default_ker(x, z):
    return x.dot(z.T)

## svm implementation
def clip(value, lower, upper):
    if value < lower:
        return lower
    if value > upper:
        return upper
    return value 
def svm_smo(x, y, ker, C, max_iter, epsilon = 1e-5):
    n, _ = x.shape
    alpha = np.zeros((n,))

    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] =  ker(x[i], x[j])
    
    iter = 0
    while iter <= max_iter:
        for i in range(n):
            j = np.random.randint(low = 0, high = n - 1)
            while (i == j):
                j = np.random.randint(low = 0, high = n - 1)
            
            eta = K[j, j] + K[i, i] - 2.0 * K[i, j]
            if np.abs(eta) < epsilon:
                continue

            e_i = (K[:, i] * alpha * y).sum() - y[i]
            e_j = (K[:, j] * alpha * y).sum() - y[j]
            alpha_i = alpha[i] - y[i] * (e_i - e_j) / eta
            
            lower, upper = 0, C
            zeta = alpha[i] * y[i] + alpha[j] * y[j]
            if y[i] == y[j]:
                lower = max(lower, zeta / y[j] - C)
                upper = min(upper, zeta / y[j])
            else:
                lower = max(lower, -zeta / y[j])
                upper = min(upper, C - zeta / y[j])
            
            alpha_i = clip(alpha_i, lower, upper)
            alpha_j = (zeta - y[i] * alpha_i) / y[j]

            alpha[i], alpha[j] = alpha_i, alpha_j
        
        iter += 1
    
    b= 0
    for i in range(n):
        if epsilon < alpha[i] < C - epsilon:
            b = y[i] - (y * alpha * K[:, i]).sum()

    def f(X):
        results = []
        for k in range(X.shape[0]):
            result = b
            for i in range(n):
                result += y[i] * alpha[i] * ker(x[i], X[k])
            result.append(result)
        return np.array(results)
    
    return f, alpha, b

## data visualization
def data_visualization(x, y):
    import matplotlib.pyplot as plt 
    category = {'+1': [], '-1': []}
    for point, label in zip(x, y):
        if label == 1.0:
            category['+1'].append(point)
        else: 
            category['-1'].append(point)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for label, pts in category.items():
        pts = np.array(pts)
        ax.scatter(pts[:, 0], pts[:, 1], label = label)
    plt.show()

def simple_synthetic_data(n, n0 = 5, n1 = 5):
    w = np.random.rand(2)
    w = w / np.sqrt(w.dot(w))

    x = np.random.rand(n, 2) * 2 - 1
    d = (np.random.rand(n) + 1) * np.random.choice([-1, 1], n, replace = True)
    d[:n0] = -1
    d[n0:n0+n1] = 1
    print(x.dot(w))

    x = x - x.dot(w).reshape(-1, 1) * w.reshape(1, 2) + d.reshape(-1, 1) * w.reshape(1, 2)

    y = np.zeros(n)
    y[d < 0] = -1
    y[d >= 0] = 1
    return x, y

np.random.seed(0)

# svm for simple synthetic data
x, y = simple_synthetic_data(100, n0 = 5, n1 = 5)

ker = default_ker
model, alpha, bias = svm_smo(x, y, ker, 1e10, 1000)

import matplotlib.pyplot as plt 
category = {'+1': [], '-1':[]}
for point, label in zip(x, y):
    if label == 1.0:
        category['+1'].append(point)
    else:
        category['-1'].append(point)
fig = plt.figure()
ax = fig.add_subplot(111)

for label, pts in category.items():
    pts = np.array(pts)
    ax.scatter(pts[:, 0], pts[:, 1], label = label)

weight = 0
for i in range(alpha.shape[0]):
    weight += alpha[i] * y[i] * x[i]

x1 = np.min(x[:, 0])
y1 = (-bias - weight[0] * x1) / weight[1]
x2 = np.max(x[:, 0])
y2 = (-bias - weight[0] * x2) / weight[1]
ax.plot([x1, x2], [y1, y2])

for i, alpha_i in enumerate(alpha):
    if abs(alpha_i) > 1e-3:
        ax.scatter([x[i, 0]], [x[i, 1]], s = 150, c = 'none', alpha = 0.7, linewidth = 1.5, edgecolor = '#FF0000')

plt.show()