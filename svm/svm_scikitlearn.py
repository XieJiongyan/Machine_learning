import numpy as np
import matplotlib.pyplot as plt
def spiral_data():
    data = np.loadtxt('input/spiral.txt')
    x = data[:, :2]
    y = data[:, 2]
    return x, y
    
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


from sklearn import svm
x, y = spiral_data()

model = svm.SVC(kernel = 'rbf', gamma = 50, tol = 1e-6)
model.fit(x, y)

fig = plt.figure(figsize = (6, 6))
ax = fig.add_subplot(111)
plot(ax, model.predict, x, 'SVM + RBF')
plt.show()

