import numpy as np
from matplotlib import pyplot as plt 

from sklearn.datasets import make_classification

#X, y: samples
X, y = make_classification(n_samples = 1000, n_features = 10, random_state = 0) 
print("The first sample of dataset:")
print('features:', X[0], 'label:', y[0])

from sklearn.tree import DecisionTreeClassifier as DTC

class RandomForest():
    def __init__(self, n_trees = 10, max_features = 'sqrt', oob_score = False): 
        self.n_trees = n_trees
        self.oob_score = oob_score 
        self.trees = [DTC(max_features = max_features) for _ in range(n_trees)]

    def fit(self, X, y):
        n_samples, _ = X.shape
        self.n_classes = np.unique(y).shape[0] 
        if self.oob_score:
            proba_oob = np.zeros((n_samples, self.n_classes))
        for tree in self.trees:
            sampled_idx = np.random.randint(0, n_samples, n_samples) 
            if self.oob_score:
                unsampled_mask = np.bincount(sampled_idx, minlength = n_samples) == 0
                unsampled_idx = np.arange(n_samples)[unsampled_mask]
            tree.fit(X[sampled_idx], y[sampled_idx])
            if self.oob_score:
                proba_oob[unsampled_idx] += tree.predict_proba(X[unsampled_idx])
        
        if self.oob_score:
            ## self.oob_score_ = np.mean(y == argmax(proba_oob, axis = 1), axis = 0)
            self.oob_score_ = np.mean(y == np.argmax(proba_oob, axis = 1))
    def predict(self, X):
        proba = self.predict_proba(X) 
        return np.argmax(proba, axis = 1) 
    def predict_proba(self, X):
        proba = np.zeros((X.shape[0], self.n_classes))
        for tree in self.trees:
            proba += tree.predict_proba(X)
        proba /= self.n_trees
        return proba
    def score(self, X, y):
        return np.mean(y == self.predict(X))

sz = 50
n_trees = np.arange(1, sz).tolist()
oob_score = np.zeros(sz)
train_score = np.zeros(sz)
for n_tree in n_trees:
    rf = RandomForest(n_trees = n_tree, oob_score = True)
    rf.fit(X, y)
    oob_score[n_tree] = rf.oob_score_
    train_score[n_tree] = rf.score(X, y)

plt.plot(n_trees, oob_score[1:], label = 'oob score')
plt.plot(n_trees, train_score[1:], label = 'train score')
plt.ylabel('score')
plt.xlabel('n_trees')
plt.legend(loc = "lower right")
plt.show()