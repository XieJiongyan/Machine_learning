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

## sklearn
from sklearn import linear_model

# suppose y = alpha x + beta
linreg = linear_model.LinearRegression() 
linreg.fit(x_train, y_train) #no need to use hstack to get another 1 vector character
print(linreg.coef_) # print alpha(vector)
print(linreg.intercept_) # print beta
y_pred_test = linreg.predict(x_test)

rmse_loss = np.sqrt(np.square(y_test - y_pred_test).mean())
print('rmse_loss:', rmse_loss)
