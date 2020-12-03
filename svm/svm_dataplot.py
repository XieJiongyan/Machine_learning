import numpy as np
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

    x = x - x.dot(w).reshape(-1, 1) * w.reshape(1, 2) + d.reshape(-1, 1) * w.reshape(1, 2)

    y = np.zeros(n)
    y[d < 0] = -1
    y[d >= 0] = 1
    return x, y

np.random.seed(0)
x, y = simple_synthetic_data(200)
data_visualization(x, y)

def spiral_data():
    data = np.loadtxt('input/spiral.txt')
    x = data[:, :2]
    y = data[:, 2]
    return x, y

x, y = spiral_data()
data_visualization(x, y)
