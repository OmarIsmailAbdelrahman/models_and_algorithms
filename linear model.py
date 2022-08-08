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
    def miniGradient(self, biasflag, batch_size, maxiter):
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

    def __init__(self, X, y, alpha=0.001, bias=False, type="Batch"):
        self.X = X
        self.y = y
        print(X.shape)
        row, col = X.shape
        self.W = np.zeros(col + 1)
        self.W[0] = 0
        self.bias = bias
        self.gradient = type
        self.alpha = alpha

    def miniGradient(self, biasflag, batch_size=32, maxiter=100):
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
            if (prev / cost).max() < 1.3 and (prev / cost).max() > 0.7:
                if tri != "p":
                    t = 0
                    tri = "p"
                t += 1
                if t > 5:
                    self.alpha *= 2
                    print("change in the alpha", self.alpha)
            else:
                if tri != "n":
                    t = 0
                    tri = "n"
                t += 1
                if t > 5:
                    self.alpha /= 2
                    print("change in the alpha", self.alpha)
            print(i, cost)
            costplotting.append(-cost)
            prev = cost
        costplotting = np.array(costplotting)
        costplotting = np.transpose(costplotting)
        newcost = costplotting[0] + costplotting[1]
        plt.scatter(np.arange(maxiter), newcost)
        return self.W

    def StGradient(self, biasflag):
        return self.W

    def train(self, batch_size=32, maxiter=30):
        if self.gradient == "stochastic":
            return self.StGradient(self.bias)
        elif self.gradient == "mini":
            return self.miniGradient(self.bias, batch_size, maxiter)
        else:
            return self.miniGradient(self.bias, batch_size, maxiter)


# Data set for linear regression
dataSize = 1000
true_slope = 13.0
true_intercept = 3.15
X = np.arange(0.0, dataSize)
random.shuffle(X)

X = np.random.randint(100, size=(1000, 300)) / 100
y = np.random.randint(10000, size=(1000)) / 100
X = np.sort(X)
y = np.sort(y)

# y = true_slope * X + true_intercept + 50 * np.random.rand(len(X))
X = X / (X.max())
y = y / X.max()

# X = np.reshape(X,(dataSize,1))
sad = linearModel(X, y)
we = sad.train(maxiter=500)
reg = LinearRegression().fit(X, y)

print()
print("model difference", sad.W[0] - reg.intercept_)
print((sad.W[1:] - reg.coef_))
print("model parameters:", sad.W)

# line = we[0] + we[1:]*X
# plt.plot(X, line, '-r', label='Batch')
# plt.scatter(X,y)
plt.xlabel("iterations")
plt.ylabel("grad")
plt.show()

# Tips:
# 1.scaling might affect gradient, because if the data have large values it might not converge because the cost function will be too large to converge
# Standardization Vs Normalization
# Neither Normalization nor Standardization changes the distribution of the data
# normalization will only scale values to certain range [normalization is sensitive to outliers, standardization is more robust to outliers]
# Standardization using  Z-score Normalization
# Standardization is more effective if the feature has a Gaussian distribution
# 2.changing the alpha while training model will help convergence a lot faster
# 3.cost function doesn't always converge to zero, because the best solution might not intersect with any data, it will have the lowest cost function
# but not equal to zero.

# conclusion: observe the data before training, it might heavily affect the model and the output, also check the distribution of data to
# try out different alpha values or use momentum

# Q: I can't find the problem here, the model in big numbers not work correctly and the cost is too big, when I scale it works fine.
# Q: i don't understand the distribution of the prior in MAP

# mathematical derivation:
#   y is y actual amd y' is predicted, and it is equal to theta*X
#   y = y' + error => error = y - y' , the error id an independent variable that have Normal distribution
#   that means that N(e) = (1/2*i*omega) * exp ^ (e/2*pi*omega)^2
#   since e is a function of x and y we can present the equation in this form:
#   (1/2*i*omega) * exp ^ ((y-theta*x)/2*pi*omega)^2 which equals to N(y|X,theta)
#   this means that the mean is the predicted y'
#   this plot a normal distribution for a given x, and mean = y'
#   from the distribution we can see the probability of actual y
#   we got the probability of one sample, using the entire dataset we get the product of every sample
#   P(dataset) = Π P(xi) =  (1/2*i*omega) * exp ^ ((y-theta*X)/2*pi*omega)^2  = L(theta)                                <==== LIKELIHOOD equation
#   maximizing the equation will result in the best theta because the probability of finding y actual is greater
#   "in maximization problem taking the log does"
#   taking the log of P "it won't affect solution because it will not change the function characteristics " will give us
#   the cost function sum of squared error
#
#   we can derived the equation for the less square error by maximizing the Likelihood
#   and using the Map maximization we get the regularization term L2
#
#
#   "The likelihood function describes the joint probability of the observed data as a function of the parameters of the chosen statistical model."
#
#   more thoughts in Likelihood:
#   the idea here is that the hypothesis create a function that predict the dependent variable y' = WX , and because the hypothesis might be wrong
#   we added to the function an error term that follow normal distribution and have mean of zero y' = WX + e , this make the dependent variable have a distribution that
#   have mean of WX, observing data -either its correct or have error-, we can find the probability using the distribution.
#   having high probability means that the hypothesis have a high probability to capture a real data point, and the same goes the other way
#   and the hypothesis space is the weights W because they are the parameters of the function y, so finding the best hypothesis "Weights" means finding
#   the hypothesis with the highest data point probability.
