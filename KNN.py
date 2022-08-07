import math

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from sklearn import datasets
from sklearn.model_selection import train_test_split

np.set_printoptions(threshold=sys.maxsize)

# set the max columns,row to none
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)





class KNN:
    # type is if distance between points matter, algo search algorithm but its brute for now"later",p for how to calculate the distance
    def __init__(self, k=5, type="uniform", algo="auto", p=2, classes=3):
        self.k = k
        self.type = type
        self.algo = algo
        self.p = p
        self.classes = classes

    def distance(self, point, df, p=2):
        dis = []
        if p == 1:
            for i in range(df.shape[0]):
                temp = df.iloc[i]
                dis.append(np.abs((point - temp).to_numpy()).sum())
            dis = (np.array(dis))
            dis = np.vstack([dis, np.array(range(1, dis.shape[0] + 1))])
            print(np.transpose(dis))
            return pd.DataFrame(data=dis, index=df.index, columns=['dis'])
        else:
            for i in range(df.shape[0]):
                temp = df.iloc[i]
                t = math.sqrt(np.power((point - temp).to_numpy(), 2).sum())
                dis.append(t)
            dis = np.array(dis)
            return pd.DataFrame(data=dis, index=df.index, columns=['distance'])

    def train(self, X, y):
       fold = math.floor(X.shape[0]/self.k)
       for i in range(k):
           validation = X.iloc[i * fold:(i + 1) * fold, :]
           newfalse = 0
           for i in range(X.shape[0]):
               if i * fold < i < (i + 1) * fold:
                   continue
               else:
                   if self.predict(X, X.iloc[i], y) != y.iloc[i]:
                       newfalse += 1
           print("score",1-newfalse/X.shape[0])


    def predict(self, df, X, y):
        dis = self.distance(X, df)
        dis = dis.sort_values(by=["distance"], axis=0)
        dis_ = np.array(dis.index)
        dis = np.array(dis)
        res = np.zeros(shape=[1, self.classes])
        if self.type == "uniform":
            for i in range(1, self.k + 1):
                res[0, y[dis_[i]]] += 1
        else:
            for i in range(1, self.k + 1):
                res[0, y[dis_[i]]] += dis[i]
        max = 0
        for i in range(res.shape[1]):
            if res[0, max] < res[0, i]:
                max = i
        return max


#    def validation(self):

iris = datasets.load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df["target"] = iris.target
x = df.drop("target", axis=1)
y = df["target"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(y)
k = KNN(classes=df.groupby(["target"]).count().shape[0], k=10, type="")
k.train(x, y)
# temp = temp.sort_values(by = ["distance"],axis=0)


# tips

# mathematical and statistical view

# problems
