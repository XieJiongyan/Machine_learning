from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import numpy as np
from matplotlib import pyplot as plt

model = Sequential()
model.add(Dense(8, input_dim=2, activation="relu"))
model.add(Dense(2, activation="softmax"))

sgd = SGD(lr=.1)

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) 
Y = np.array([[0, 1], [1, 0], [1, 0], [0, 1]]) 

model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X, Y, epochs=200)