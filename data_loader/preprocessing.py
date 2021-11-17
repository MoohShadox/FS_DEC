
import re

import numpy as np
import pandas as pd
from sklearn import preprocessing

def one_hot_encode_column(a):
    b = np.zeros((a.size, a.max() + 1))
    b[np.arange(a.size), a] = 1
    return b

def one_hot_encoding(a, categorical_indexes):
    arr = np.array(a)
    L = []
    for i in range(arr.shape[1]):
        if(i in categorical_indexes):
            enc = preprocessing.OneHotEncoder()
            a = enc.fit_transform(arr[:, i].reshape((-1, 1))).toarray()
            for k in range(a.shape[1]):
                L.append(a[:, k])
        else:
            L.append(arr[:, i])
    L = np.array(L).T
    return L
