# missing features
# bias can be removed
# gradient checking
# cross-validation test for lambda constant
import sys
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt

np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

df = pd.read_csv("train.csv")

# preprocessing
def preprocess(X):
    X['Fare'].fillna(int(X['Fare'].mean()), inplace=True)
    X['Age'].fillna(int(X['Age'].mean()), inplace=True)
    X = X.to_numpy()
    for i in range(X.shape[1]):
        X[:, i] = (X[:, i] - X[:, i].mean()) / X[:, i].std()
    # print(X.shape)
    return X

X = df.drop(["Name", "Ticket", "Cabin", "Embarked", "PassengerId"], axis=1)
X.replace({'male': 1, 'female': 0}, inplace=True)
X = preprocess(X)
y = np.array(df["Survived"]).reshape(X.shape[0], -1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

class NN:

    def __init__(self, hidden_layers, output_layer,learning_rate=0.001,max_iter=1000, reg_constant = 0.001,bias= True):
        self.max_iter = max_iter
        self.hidden_layers = hidden_layers
        self.output_layer = output_layer
        self.learning_rate = learning_rate
        self.reg_constant = reg_constant
        self.bias = bias
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def dsigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def loss(self, y, y_hat):
        return 1 / 2 * (y - y_hat) ** 2

    def dloss(self, y, y_hat):
        return (y - y_hat)

    def layers(self, size, input_size):
        if self.hidden_layers != size.shape[0]:
            print("Wrong number of layers")
            return
        # creating weights for each layer
        W = []
        np.random.seed(1234)
        W.append(np.random.rand(input_size + 1, size[0]))
        for i in range(len(size)):
            np.random.seed(1235+i)
            if i == 0:
                continue
            W.append(np.random.rand(size[i - 1] + 1, size[i]))
        np.random.seed(12)
        W.append(np.random.rand(size[len(size) - 1] + 1, self.output_layer))
        self.W = W
        for i in range(len(W)):
            print("layer number ", i, "shape", W[i].shape)

    def forward(self, X, y):
        A = []
        A.append(X)
        X = np.hstack((np.ones([X.shape[0], 1]), X))
        Z = []
        Z.append(np.dot(X , self.W[0]))
        A.append(self.sigmoid(Z[0]))
        # print(Z[0].shape,X.shape,self.W[0].shape)
        for i in range(self.hidden_layers):
            matrix = self.W[i + 1]
            tmp = A[len(A) - 1]
            tmp = np.hstack((np.ones([tmp.shape[0], 1]), tmp))
            Z.append(np.dot(tmp, matrix))
            A.append(self.sigmoid(Z[len(Z) - 1]))
        # print(Z[len(Z) - 1])
        self.Z = Z
        self.A = A
        self.backward(y)

    def avg_bias(self,db):
        db = np.array(db).mean(axis = 0)
        return db

    def backward(self, y):
        dc = (self.A[len(self.A) - 1] - y) * 2
        for i in range(self.hidden_layers + 1):
            if i == 0:
                db = self.dsigmoid(self.Z[-(1+i)]) * dc
                dw= np.dot(self.A[-(2+i)].T,db)
            else:
                db = np.dot(self.dsigmoid(self.Z[-(1+i)]),np.dot(self.W[-i][1:],dw.T))
                dw = np.dot(self.A[-(2+i)].T,db)

            db = self.avg_bias(db)
            reg1 = self.W[-(1 + i)][0]
            reg2 = self.W[-(1+i)][1:]
            self.W[-(1+i)][0] = self.W[-(1+i)][0] - (db + np.sqrt(np.power(reg1,2).sum()) * self.reg_constant)*(self.learning_rate/X.shape[0])
            self.W[-(1+i)][1:] = self.W[-(1+i)][1:] - (dw + np.sqrt(np.power(reg2,2).sum()) * self.reg_constant )*(self.learning_rate/X.shape[0])

    def train(self,X,y,batch_size=100,):
        for i in range(self.max_iter):
            itr = int(X.shape[0]/batch_size + 1)
            for j in range (itr):
                s = (j*batch_size)%X.shape[0]
                e = ((j+1)*batch_size)%X.shape[0]
                if s > e:
                    e = X.shape[0]
                self.forward(X[s:e],y[s:e])

    def predict(self,X):
        A = []
        A.append(X)
        X = np.hstack((np.ones([X.shape[0], 1]), X))
        Z = []
        Z.append(np.dot(X, self.W[0]))
        A.append(self.sigmoid(Z[0]))
        for i in range(self.hidden_layers):
            matrix = self.W[i + 1]
            tmp = A[len(A) - 1]
            tmp = np.hstack((np.ones([tmp.shape[0], 1]), tmp))
            Z.append(np.dot(tmp, matrix))
            A.append(self.sigmoid(Z[len(Z) - 1]))
        return A[-1]

model = NN(hidden_layers=2, output_layer=1,learning_rate=1,max_iter=1000,reg_constant= 0.01)
model.layers(size=np.array([8,4]), input_size=X.shape[1])
model.train(X_train,y_train)
#test
predicted = np.where( model.predict(X_test) > 0.5, 1, 0)
f = 0
for i in range(y_test.shape[0]):
    if y_test[i] != predicted[i]:
        f+=1
print("score ", (1- f/X.shape[0]), " number of test cases" , X.shape[0])
print(f)



# Tips:
# 1. You can usually prevent over-fitting if you keep your number of neurons below:
#       size of the data set "rows" / alpha * ( size of the input layer + size of the output layer)
# where alpha is number between 2 - 10
# 2. Regularization technique:
#       1.Dropout   2.L2,L1
# 3. Optimization of the Network Configuration:
#    1.pruning nodes


