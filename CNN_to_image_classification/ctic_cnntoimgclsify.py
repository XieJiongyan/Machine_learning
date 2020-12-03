from keras.datasets import cifar10 

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

num_classes = 10
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

import numpy as np 
import matplotlib.pyplot as plt 

from keras.utils import to_categorical 

x_train = x_train / 255 
x_test = x_test / 255 

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes) 
print(y_train.shape) 
print(y_train[0]) 

from keras.models import Sequential 
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout 

model = Sequential() 

model.add(Conv2D(32, (3, 3), padding = "same", input_shape = x_train.shape[1:], activation = "relu"))
model.add(MaxPooling2D(pool_size = 2)) 
model.add(Dropout(0.25)) 

model.add(Flatten()) 
model.add(Dense(512, activation = "relu")) 
model.add(Dropout(0.5)) 
model.add(Dense(num_classes, activation = "softmax")) 

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy']) 
model.summary() 

batch_size = 32 
epochs = 5 
history = model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, verbose = 1, validation_data = (x_test, y_test)) 
score = model.evaluate(x_test, y_test, verbose = 0) 
print('Test loss:', score[0]) 
print('Test accuracy:', score[1])
