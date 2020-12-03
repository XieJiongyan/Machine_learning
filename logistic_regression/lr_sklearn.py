## data preparation
import numpy as np
import matplotlib.pyplot as plt 
from sklearn import linear_model

lines = np.loadtxt('input/logisticR_data.csv', delimiter = ',', dtype = 'str')
x_total = lines[:, 1:3].astype('float')
y_total = lines[:, 3].astype('float')

pos_index = np.where(y_total == 1)
neg_index = np.where(y_total == 0)
plt.scatter(x_total[pos_index, 0], x_total[pos_index, 1], marker = 'o', c = 'r')
plt.scatter(x_total[neg_index, 0], x_total[neg_index, 1], marker = 'x', c = 'b')
print('data_set size', y_total.shape[0])

## sklearn

lr_clf = linear_model.LogisticRegression()
lr_clf.fit(x_total, y_total)
print(lr_clf.coef_[0])
print(lr_clf.intercept_)
y_pred = lr_clf.predict(x_total)
print('accurancy:', (y_pred == y_total).mean())

plt.show()
