import re

import numpy as np
import pandas as pd
from sklearn import preprocessing

from data_loader.preprocessing import one_hot_encoding


def loadGerman():
    f = open("/home/moohshadox/PycharmProjects/FEDEC/data_loader/datasets/german.data")
    L = f.readlines()
    arr = []
    for i in L:
        ch = (i.strip())
        ch = [float(i) for i in re.findall("\d+", ch)]
        arr.append(ch)
    arr = np.array(arr)
    arr = one_hot_encoding(arr, [0,2,3,5,8,9,11,13,14,16])
    X = arr[:,:-1]
    y = arr[:,-1]
    return X, np.where(y == 2, 1 , 0)


