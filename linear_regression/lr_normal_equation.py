# -*- coding: utf-8

import numpy as np
from sklearn import preprocessing
import os

# load data
lines = np.loadtxt('input/USA_Housing.csv', delimiter = ',', dtype = 'str')
print("??")
for i in range(lines.shape[1]-1):
    print(lines[0, i])
print("??")
print(lines[0, -1])

x_total = lines[1:, :5].astype('float')
y_total = lines[1:, 5:].astype('float').flatten()

x_total = preprocessing.scale(x_total)
y_total = preprocessing.scale(y_total)
mid = 4000 # manually
x_train = x_total[:mid]
x_test = x_total[mid:]
y_train = y_total[:mid]
y_test = y_total[mid:]
print('train_set', x_train.shape[0])
print('test_set', x_test.shape[0])

# normal equation
X_train = np.hstack([x_train, np.ones((x_train.shape[0], 1))]) # ????
NE_solution = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X_train), X_train)), np.transpose(X_train)), y_train.reshape([-1, 1]))

print(NE_solution)
X_test = np.hstack([x_test, np.ones((x_test.shape[0], 1))])
y_pred_test = np.dot(X_test, NE_solution).flatten()
rmse_loss = np.sqrt(np.square(y_test - y_pred_test).mean())
print('rmse_loss:', rmse_loss)
print("a") 