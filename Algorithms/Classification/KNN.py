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
            return pd.DataFrame(data=dis, index=df.index, columns=['dis'])
        else:
            for i in range(df.shape[0]):
                temp = df.iloc[i]
                t = math.sqrt(np.power((point - temp).to_numpy(), 2).sum())
                dis.append(t)
            dis = np.array(dis)
            return pd.DataFrame(data=dis, index=df.index, columns=['distance'])

    def train(self, X, y):
        fold = math.floor(X.shape[0] / self.k)
        trainscore = 0
        validationscore = 0
        for j in range(self.k):
            validation = X.iloc[j * fold:(j + 1) * fold, :]
            newfalse = 0
            for i in range(X.shape[0]):
                if (j * fold) < i < ((j + 1) * fold):
                    continue
                else:
                    if self.predict(X, X.iloc[i], y) != y.iloc[i]:
                        newfalse += 1
            trainscore += 1 - newfalse / X.shape[0]

            for i in range(validation.shape[0]):
                if self.predict(X, validation.iloc[i], y) != y.iloc[i]:
                    newfalse += 1
            validationscore += 1 - newfalse / X.shape[0]
        print(trainscore / self.k, validationscore / self.k)
        return trainscore / self.k

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

    def test(self,df,X,y,y_test):
        fault = 0
        for i in range(X.shape[0]):
            if self.predict(df, X.iloc[i], y) != y_test.iloc[i]:
                fault += 1
        print(1-(fault/X.shape[0]))
        return 1-(fault/X.shape[0])


# iris = datasets.load_iris()
# df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
# df["target"] = iris.target
# x = df.drop("target", axis=1)
# y = df["target"]

# df0 = df.loc[df['target'].isin([0])]
# df1 = df.loc[df['target'].isin([1])]
# df2 = df.loc[df['target'].isin([2])]

df = pd.read_csv("diabetes.csv")
x = df.drop("Outcome",axis=1)
y = df["Outcome"]


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)
training = []
testing = []
t = 10
for i in range(1,t+1):
    print(i)
    k = KNN(classes=df.groupby(["Outcome"]).count().shape[0], k=i, type="uniform",p=1)
    training.append( k.train(x, y))
    testing.append(k.test(x_train,x_test,y_train,y_test))
# temp = temp.sort_values(by = ["distance"],axis=0)
print(np.array(training),np.array(testing))
plt.scatter(np.array(range(t)),np.array(training),color= "blue")
plt.scatter(np.array(range(t)),np.array(testing),color = "red")
plt.show()

# personal view
# KNN is one of the simplest algorithms, its fully depend on distance and k
# but there is tricky party because you might think it doesn't need trainin
# the training in knn model is for "FINDING THE BEST K", and this can only be made by training the model and find the best k value
# but there is a problem that we can't test if the

# tips
# 1. overfitting occure with low k value, underfitting occure with high k value, because with low k "example = 1" it will fit the data 100%, but testing the data will
# have a very bad accuracy because it consider only the nearest one and that might be, in the other hand high k will create underfitting because it might
# consider far points that might give different answer, so the training accuracy is low
# 2.sacling data is important when using weighted points
# 3.KNN model can't be regularized because it doesn't have a parameter to penalize, solving overfitting in KNN by using higher K,  A similar approach is used in decision trees

# mathematical and statistical view
# underfitting oh h: max(0 , max testAccuracy(h') - trainAccuracy(h))
# underfitting of h: max(0 , trainAccuracy(h) - testAccuracy(h))

# problems
# it's very slow for some reason everny on data 800 * 8, it took 1 hour to train