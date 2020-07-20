import numpy as np

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
            results.append(result)
        return np.array(results)
    
    return f, alpha, b

## spiral data set
def spiral_data():
    data = np.loadtxt('input/spiral.txt')
    x = data[:, :2]
    y = data[:, 2]
    return x, y

def default_ker(x, z):
    return x.dot(z.T)

def poly_ker(d):
    def ker(x, z):
        return (x.dot(z.T)) ** d
    return ker

def cos_ker(x, z):
    return x.dot(z.T) / np.sqrt(x.dot(x.T)) / np.sqrt(z.dot(z.T))

def rbf_ker(sigma): # rbf kernel
    def ker(x, z):
        return np.exp(-(x - z).dot((x - z).T) / (2.0 * sigma ** 2))
    return ker

import matplotlib.pyplot as plt
from matplotlib import cm

def plot(ax, model, x, title):
    y = model(x)
    y[y < 0], y[y >= 0] = -1, 1

    category = {'+1': [], '-1': []}
    for point, label in zip(x, y):
        if label == 1.0:
            category['+1'].append(point)
        else:
            category['-1'].append(point)
    for label, pts in category.items():
        pts = np.array(pts)
        ax.scatter(pts[:, 0], pts[:, 1], label = label)
    
    p = np.meshgrid(np.arange(-1.5, 1.5, 0.025), np.arange(-1.5, 1.5, 0.025))
    x = np.array([p[0].flatten(), p[1].flatten()]).T
    y = model(x)
    y[y < 0], y[y >= 0] = -1, 1
    y = np.reshape(y, p[0].shape)
    ax.contourf(p[0], p[1], y, cmap = plt.cm.coolwarm, alpha = 0.4)

    ax.set_title(title)

fig = plt.figure(figsize = (12, 6))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

x, y = spiral_data()

model_default, _, _ = svm_smo(x, y, default_ker, 1e10, 200)
plot(ax1, model_default, x, 'Default SVM')

ker = rbf_ker(0.2)

model_ker, _, _ = svm_smo(x, y, ker, 1e10, 200)
plot(ax2, model_ker, x, 'SVM + RBF')

plt.show()