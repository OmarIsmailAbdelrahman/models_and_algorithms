# multivariable linear regression
# this will contain linear regression using
import random

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sys
np.set_printoptions(threshold=sys.maxsize)
class model(ABC):
    @abstractmethod
    def miniGradient(self, biasflag,batch_size,maxiter):
        pass

    @abstractmethod
    def StGradient(self, biasflag):
        pass
class linearModel(model):
    X = []
    y = []
    W = []
    bias = False
    gradient = "Batch"
    alpha = 0.001
    def __init__(self,X,y,alpha = 0.001 , bias=False, type="Batch"):
        self.X = X
        self.y = y
        print(X.shape)
        row, col = X.shape
        self.W = np.zeros(col+1)
        self.W[0] = 0
        self.bias = bias
        self.gradient = type
        self.alpha = alpha

    def miniGradient(self, biasflag,batch_size = 32,maxiter = 100):
        self.X = np.hstack((np.zeros((self.X.shape[0], 1)) + 1, self.X))
        t = 0
        tri = ""
        prev = 1

        costplotting = []

        for i in range(maxiter):
           h = np.dot(self.X, self.W)
           cost = h - y
           cost = np.dot(np.transpose(self.X), cost)
           self.W = self.W - cost * (self.alpha / self.X.shape[0])
           if (prev/cost).max() < 1.3 and (prev/cost).max() > 0.7 :
               if tri != "p":
                   t = 0
                   tri = "p"
               t+=1
               if t > 5:
                self.alpha *= 2
                print("change in the alpha",self.alpha)
           else:
               if tri != "n":
                   t = 0
                   tri = "n"
               t+=1
               if t > 5:
                self.alpha /= 2
                print("change in the alpha",self.alpha)
           print(i, cost)
           costplotting.append(-cost)
           prev = cost
        costplotting = np.array(costplotting)
        costplotting = np.transpose(costplotting)
        newcost = costplotting[0] + costplotting[1]
        plt.scatter(np.arange(maxiter),newcost)
        return self.W

    def StGradient(self, biasflag):
        return self.W

    def train(self,batch_size=32,maxiter=30):
        if self.gradient == "stochastic":
            return self.StGradient(self.bias)
        elif self.gradient == "mini":
            return self.miniGradient(self.bias,batch_size,maxiter)
        else:
            return self.miniGradient(self.bias,batch_size,maxiter)





# Data set for linear regression
dataSize = 1000
true_slope = 13.0
true_intercept = 3.15
X = np.arange(0.0, dataSize)
random.shuffle(X)


X = np.random.randint(100,size = (1000,300))/100
y = np.random.randint(10000,size = (1000))/100
X = np.sort(X)
y = np.sort(y)


#y = true_slope * X + true_intercept + 50 * np.random.rand(len(X))
X =X/(X.max())
y = y/y.max()

#X = np.reshape(X,(dataSize,1))
sad = linearModel(X,y)
we = sad.train(maxiter=5000)
reg = LinearRegression().fit(X, y)

print()
print("model difference", sad.W[0]-reg.intercept_)
print((sad.W[1:]-reg.coef_))
print("model parameters:",sad.W)



#line = we[0] + we[1:]*X
#plt.plot(X, line, '-r', label='Batch')
#plt.scatter(X,y)
plt.xlabel("iterations")
plt.ylabel("grad")
plt.show()






#Tips:
#1.scaling might effect gradient, becuase if the data have large values it might not converge because the cost function will be too large to converge
#Standardization Vs Normalization
#Neither Normalization nor Standardization changes the distribution of the data
#normalization will only scale values to certain range [normalization is sensitive to outliers, standardization is more robust to outliers]
#Standardization using  Z-score Normalization
#Standardization is more effective if the feature has a Gaussian distribution

###conclution: observe the data before training, it might heavely effect the model and the output, also check the distribution of data too

#2.changing the alpha while training model will help convergence a lot faster


#Q:##i cant find the problem here, the model in big numbers not work correctly and the cost is too big, when i sacle it works fine.