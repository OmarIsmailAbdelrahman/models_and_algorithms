import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import sys
np.set_printoptions(threshold=sys.maxsize)

#normaly distruction data
mean = 0
sd = 7
x = np.random.normal(mean,sd, 1000)

#sin function
# t = []
# for i in range(s.shape[0]):
#     t.append( math.sin(y[i]))
# t = np.array(t) + 0.5* np.random.rand(len(s))
# y = y + 5 * np.random.rand(len(s))
# print(t,t.shape)

#Curve for polynomial 2x² -5x + 4 = 0
y =  2*np.power(x,2) + (-5 * x) + 4
#Curve for polynomial 2x² -5x + 4 = 0
# mean = 0
# sd = 5
# s = np.random.normal(mean,sd, 100)
#
# x = s
# y =  2* x*x -5 * x + 4 +   100* np.random.rand(len(s))







class PolynomailRegression():
    degree = 3
    alpha = 0.0001
    iterations = 200
    def __int__(self,degree = 2,alpha = 0.000000000000001,iterations = 200):
        self.degree = degree
        self.alpha = alpha
        self.iterations = iterations
    def train(self,X,y,degree = 2,iterations = 1000):
        #weight intilization
        prev = 0
        num = 0
        t=0
        tri = ""
        W = np.zeros(degree+1)
        #cost function calculation and updating weights
        for i in range(iterations):
            cost = 0
            for j in range(degree+1):
                z = np.array(X)
                z = np.power(z,j)
                cost+= np.dot(z,W[j])
            cost = ( y-cost)
            print("cost",cost.max())
            prev = cost
            for j in range(degree+1):
                temp = np.array(X)
                temp = np.power(temp,j)
                temp = np.transpose(temp)
                sad = np.dot(temp,cost)*(self.alpha/X.shape[0])
                print(j,'    ', sad)
                W[j] = W[j] + sad
            print("w", W)
            print()
            #print(i,W)
        return W
            #print(i,cost)


model = PolynomailRegression()
print(x)
#print(x)
weights = model.train(x,y,iterations = 500)

y_ = weights[2] *np.power(x,2) + (weights[1] * x) + weights[0]
print("check result",y_-y)
plt.scatter(x,y)
plt.scatter(x,y_)
plt.show()