# -*- coding: utf-8

import numpy as np
from sklearn import preprocessing
import os
import matplotlib.pyplot as plt 

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

# gradient descent

def shuffle_aligned_list(data):
    num = data[0].shape[0]
    shuffle_index = np.random.permutation(num)
    return [d[shuffle_index] for d in data]

def batch_generator(data, batch_size, shuffle = True):
    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size >= data[0].shape[0]:
            batch_count = 0
            if shuffle:
                data = shuffle_aligned_list(data)
        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield [d[start:end] for d in data]


num_steps = 600
learning_rate = 0.01
batch_size = 40

weight = np.zeros(6)
np.random.seed(0)
batch_g = batch_generator([x_train, y_train], batch_size, shuffle = True)
x_test_concat = np.hstack([x_test, np.ones([x_test.shape[0], 1])])

loss_list = []
for i in range(num_steps):
    rmse_loss = np.sqrt(np.square(np.dot(x_test_concat, weight) - y_test).mean())
    loss_list.append(rmse_loss)
    x_batch, y_batch = batch_g.__next__()
    x_batch = np.hstack([x_batch, np.ones([batch_size, 1])])
    y_pred = np.dot(x_batch, weight)
    w_gradient = (x_batch * np.tile((y_pred - y_batch).reshape([-1, 1]), 6)).mean(axis = 0)
    weight = weight - learning_rate * w_gradient

print('weight:', weight)
print('rmse_loss:', rmse_loss)

loss_array = np.array(loss_list)
plt.plot(np.arange(num_steps), loss_array)
plt.show()