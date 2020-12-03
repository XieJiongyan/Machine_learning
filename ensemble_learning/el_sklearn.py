import numpy as np
from matplotlib import pyplot as plt 

from sklearn.datasets import make_classification
X, y = make_classification(n_samples = 1000, n_features = 10, random_state = 0) 
print("The first sample of dataset:")
print('features:', X[0], 'label:', y[0])

from sklearn.ensemble import BaggingClassifier 
from sklearn.tree import DecisionTreeClassifier as DTC

bc = BaggingClassifier(DTC(max_features = 'sqrt'), n_estimators = 10, oob_score = True) 
bc.fit(X, y)
print('oob_score:', bc.oob_score_)