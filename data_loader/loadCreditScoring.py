import pandas as pd

def loadCreditScoringDS():
    df1 = pd.read_csv("/home/moohshadox/PycharmProjects/FEDEC/data_loader/datasets/cs-test.csv", index_col = False)
    df2 = pd.read_csv("/home/moohshadox/PycharmProjects/FEDEC/data_loader/datasets/cs-training.csv", index_col=False)
    df = pd.concat([df1, df2])
    df = df.drop(columns=[df.columns[0]]).dropna()
    arr = df.values
    X,y = arr[:, 1:], arr[:, 0]
    return X,y
