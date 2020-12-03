from keras.datasets import cifar10 

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

num_classes = 10
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

import numpy as np 
import matplotlib.pyplot as plt 
fig, axes = plt.subplots(num_classes, 10, figsize = (15, 15)) 

# visualization
for i in range(num_classes): 
    indice = np.where(y_train == i)[0] 
    for j in range(10): 
        axes[i][j].imshow(x_train[indice[j]]) 
        axes[i][j].set_xticks([])
        axes[i][j].set_yticks([]) 
plt.tight_layout() 
plt.show() 