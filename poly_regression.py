import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import sys
np.set_printoptions(threshold=sys.maxsize)

#normaly distruction data
# mean = 10
# sd = 700
# x = np.random.normal(mean,sd, 1500)
x = np.arange(0.0, 500)-250
#sin function
# t = []
# for i in range(s.shape[0]):
#     t.append( math.sin(y[i]))
# t = np.array(t) + 0.5* np.random.rand(len(s))
# y = y + 5 * np.random.rand(len(s))
# print(t,t.shape)

#Curve for polynomial 2x² +500x + 0 = y
a = 2
b = 5
c = 2000
d = 500
e = 300
f = -5
y =  a* np.power(x,5)+ b*np.power(x,4) + (c * np.power(x,3)) + (d*np.power(x,2)) + (e*np.power(x,1)) + f
#Curve for polynomial 2x² -5x + 4 = 0
# mean = 0
# sd = 5
# s = np.random.normal(mean,sd, 100)
#
# x = s
# y =  2* x*x -5 * x + 4 +   100* np.random.rand(len(s))






class PolynomailRegression:
    degree=2
    alpha=0.001
    iterations=200
    def __init__(self,degree = 2,alpha = 0.0001,iterations = 200):
        self.degree = degree
        self.alpha = alpha
        self.iterations = iterations

    def transformation(self, X):
        transformation = np.ones(X.shape[0])
        for i in range(self.degree):
            temp = np.power(X, i + 1)
            transformation = np.vstack((transformation, temp))
        return transformation
    def train(self,X,y,):
        prev = 0
        self.t=0
        self.tri =""
        trans = self.transformation(X)
        trans = np.transpose(trans)
        #weights initialization
        w = np.zeros(self.degree+1)
        for i in range(self.iterations):
           cost = ( np.dot(trans,w) - y)
           prev = cost
           w = w - np.dot(np.transpose(trans), cost)*self.alpha*(1/X.shape[0])
           print(i,w)
        return w


model = PolynomailRegression(iterations = 8000,degree=5,alpha=0.000000000000000000000005)
weights = model.train(x,y)
# y_ = weights[2] *np.power(x,2) + (weights[1] * x) + weights[0]
#Curve for polynomial 2x² -5x + 4 = 0
print(x.shape,weights.shape)
y_ = 0
for i in range(weights.shape[0]):
    y_ = y_ + weights[i]*np.power(x,i)
print("real",f,e,d,c,b,a)
print("hypothesis",weights)
print("difference",f-weights[0],e-weights[1], d-weights[2],c-weights[3],b-weights[4],a-weights[5])
print("check result",(y_-y).max())
plt.scatter(x,y_)
plt.scatter(x,y)
plt.show()