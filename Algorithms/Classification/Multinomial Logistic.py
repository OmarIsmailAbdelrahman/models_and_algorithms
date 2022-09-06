# under construction
import sys
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

np.set_printoptions(threshold=sys.maxsize)
onehot_encoder = OneHotEncoder(sparse=False)


class logistic:
    def __init__(self, alpha=0.01, multi_class="", max_iter=1000, penalty="", k=2):
        self.W = None
        self.alpha = alpha
        self.multi_class = multi_class
        self.max_iter = max_iter
        self.penalty = penalty
        self.k = k
        self.W = 0

    def softmax(self, z, Base=math.e):
        #create matrix [i*K]
        tmp = np.exp(-z)
        tmp2 = np.zeros(tmp.shape)
        for i in range(tmp.shape[0]):
            sum = 0
            for j in range(tmp.shape[1]):
                sum += tmp[i,j]
            tmp2[i,:] = tmp[i,:]/sum
        return tmp2

    def predict(self, X, c):
        # Z = Wx
        z = X.dot(self.W)

        # class probability
        t = self.softmax(z[:, c])

        # normalization constant
        sum = 0
        for i in range(self.k):
            sum += self.softmax(z[:, i])
        return t / sum

    def predictC(self, X):
        # initialize type and predict data for every row ad class
        types = np.zeros(X.shape[0])
        p = np.zeros(X.shape[0])
        z = X.dot(self.W)
        print("z",z)
        t = self.softmax(z)
        print("start testing",t)
        # return the highest proability class
        for i in range(X.shape[0]):
            for j in range(self.k):
                if t[i, j] == t[i, :].max():
                    types[i] = j
                    p[i] = t[i, j]
                    break

        return types, p

    def standrization(self, X):
        temp = np.ones(shape=X.shape)
        for i in range(1, X.shape[1]):
            # print(i,X[:,i])
            temp[:, i] = (X[:, i] - np.mean(X[:, i])) / np.std(X[:, i])
        return temp

    def grad(self, X, y):
        h = self.softmax(X.dot(self.W))
        cost = y-h
        self.W = self.W - X.T.dot(cost) * self.alpha

    def train(self, X, y):

        # encode y to matrix
        y_hot = onehot_encoder.fit_transform(y.reshape(-1, 1))
        self.k = y_hot.shape[1]

        # initialize W
        self.W = np.random.rand(X.shape[1], y_hot.shape[1])

        # standardize data
        X = self.standrization(X)
        for i in range(self.max_iter):

            # creating batches of data
            s1 = min(((i * int(X.shape[0] / 10)) % X.shape[0]), (((i + 1) * int(X.shape[0] / 10)) % X.shape[0]))
            s2 = max(((i * int(X.shape[0] / 10)) % X.shape[0]), (((i + 1) * int(X.shape[0] / 10)) % X.shape[0]))
            if s1 < X.shape[0] / 10:
                s1 = s2 - int(X.shape[0] / 10)
            batch = X[s1:s2, :]
            batchy = y_hot[s1: s2]
            self.grad(batch, batchy)
        print(X.shape,self.W.shape)

df = pd.read_csv("Iris.csv")
df["Species"] = df.apply(LabelEncoder().fit_transform)["Species"]
X = df.drop(["Species", "Id"], axis=1).to_numpy()
y = df["Species"].to_numpy()
print("start")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = logistic(k=3,alpha=0.001,max_iter=10000)
model.train(X_train, y_train)
print(model.W)
print("test model",model.predictC(X_test)[0],y_test)


