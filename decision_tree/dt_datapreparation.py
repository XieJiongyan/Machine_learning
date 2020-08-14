import numpy as np 
import matplotlib.pyplot as plt 
from scipy.sparse import csr_matrix 

CAT_FEAT = [0, 2, 4, 5, 9] 
NUM_FEAT = [3, 7] 

def parse_feat(line): 
    quota = False 
    j = 0 
    feats = []
    for i in range(len(line)): 
        if line[i] == '\"': 
            quota = not quota 
        if line[i] == ',' and not quota:
            feat = line[j:i] 
            feats.append(feat)
            j = i + 1 
    return feats + [line[j:]] 
def load_file(file_name): 
    data = [] 
    with open(file_name, 'r') as fin:
        print('field_names:', fin.readline().strip().split(','))
        for line in fin: 
            line = line.strip() 
            data.append(parse_feat(line)) 
    return np.array(data) 

train_data = load_file('input/Titanic/train.csv')
# print(train_data[1, :])
test_data = load_file('input/titanic/test.csv')

train_id, train_label, train_feat = train_data[:, 0], train_data[:, 1], train_data[:, 2:] 
test_id, test_feat = test_data[:, 0], test_data[:, 1:] 

print('train_feat:\n', train_feat[0]) 
print('test_feat:\n', test_feat[0]) 

def num2cat(num_feat, n_class = 10): 
    def to_float(x): 
        if len(x): 
            return float(x) 
        else:
            return -1 
    num_feat = np.array([to_float(x) for x in num_feat]) 
    min_val, max_val = num_feat[num_feat > -1].min(), num_feat.max() 
    sep = np.linspace(min_val, max_val, n_class, endpoint = False)
    print(sep) 
    plt.hist(num_feat[num_feat > -1], bins = n_class) 
    plt.show() 
    def indicator(x): 
        x = to_float(x) 
        if x == -1: 
            return 0
        for i in range(len(sep)):
            if x < sep[i]:
                return i 
        return n_class
    return indicator 

for nf in NUM_FEAT: 
    ind = num2cat(list(train_feat[:, nf]) + list(test_feat[:, nf])) 
    for _ in range(len(train_feat[:, nf])): 
        train_feat[_, nf] = str(ind(train_feat[_, nf])) 
    for _ in range(len(test_feat[:, nf])):
        test_feat[_, nf] = str(ind(test_feat[_, nf])) 

train_feat = np.delete(train_feat, [1, 6, 8], axis = 1) 
test_feat = np.delete(test_feat, [1, 6, 8], axis = 1) 
print('train_feat:\n', train_feat)
print('test_feat:\n', test_feat) 

class Dataset: 
    @staticmethod
    def build_feat_map(cat_feats): 
        feat_map = {} 
        for i in range(cat_feats.shape[1]):
            for x in cat_feats[:, i]: 
                feat_name = str(i) + ':' + x
                if feat_name not in feat_map:
                    feat_map[feat_name] = len(feat_map) 
        return feat_map 
    def feat2id(self, cat_feats): 
        feat_ids = [] 
        for i in range(cat_feats.shape[1]): 
            feat_ids.append([]) 
            for x in cat_feats[:, i]:
                feat_name = str(i) + ':' + x 
                feat_ids[-1].append(self.feat_map[feat_name])
        return np.int32(feat_ids).transpose() 
    def split_train_valid(self): 
        np.random.seed(0) 
        rnd = np.random.random(len(self.train_label)) 
        self.train_ind = np.where(rnd < 0.8)[0] 
        self.valid_ind = np.where(rnd >= 0.8)[0] 
        def to_csr(data, dim = len(self.feat_map)): 
            row = np. zeros_like(data) + np.expand_dims(np.arange(len(data)), 1) 
            val = np.ones_like(data)
            return csr_matrix((val.flatten(), (row.flatten(), data.flatten())), shape = (len(data), dim))
        self.train_data = (self.train_label[self.train_ind], to_csr(self.train_feat[self.train_ind])) 
        self.valid_data = (self.train_label[self.valid_ind], to_csr(self.train_feat[self.valid_ind])) 
        self.test_data = (np.zeros(len(self.test_feat), dtype = np.int32), to_csr(self.test_feat)) 
    
    def __init__(self): 
        self.feat_map = self.build_feat_map(np.vstack([train_feat, test_feat])) 
        self.train_id, self.test_id = train_id, test_id 
        self.train_label = np.int32(train_label) 
        self.train_feat, self.test_feat = self.feat2id(train_feat), self.feat2id(test_feat) 
        print('train_feat:\n', self.train_feat)
        print('test_feat:\n', self.test_feat) 
        self.split_train_valid() 

Data = Dataset()

train_label, train_feat = Data.train_data[0], Data.train_data[1].toarray() 
valid_label, valid_feat = Data.valid_data[0], Data.valid_data[1].toarray() 

print('train_feat:\n', train_feat) 
print('valid_feat:\n', valid_feat) 
