import numpy as np

from data_loader.preprocessing import one_hot_encoding


def loadAustralian():
    f = open("/home/moohshadox/PycharmProjects/FEDEC/data_loader/datasets/australian.dat")
    L = f.readlines()
    arr = []
    for i in L:
        arr.append(list(map(float,i.strip().split(" "))))
    arr = np.array(arr)
    arr = one_hot_encoding(arr, [0, 3, 4, 5, 11])
    X = arr[:,:-1]
    y = arr[:,-1]
    return X,y




