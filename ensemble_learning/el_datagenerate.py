import numpy as np
from matplotlib import pyplot as plt 

from sklearn.datasets import make_classification
X, y = make_classification(n_samples = 1000, n_features = 10, random_state = 0) 
print("The first sample of dataset:")
print('features:', X[0], 'label:', y[0])
