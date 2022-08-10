import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import sys

np.set_printoptions(threshold=sys.maxsize)

# normaly distruction data
# mean = 0
# sd = 10
# x = np.random.normal(mean,sd, 10)
x = np.arange(0.0, 2000) - 1000
# sin function
# t = []
# for i in range(s.shape[0]):
#     t.append( math.sin(y[i]))
# t = np.array(t) + 0.5* np.random.rand(len(s))
# y = y + 5 * np.random.rand(len(s))
# print(t,t.shape)
# Curve for polynomial 2x² +500x + 0 = y
a = 0.00000000001
b = 0
c = 0
d = 0
e = -5
f = 0
y = a * np.power(x, 5) + (e * x)
plt.scatter(x, y)
plt.show()


# y = a * np.power(x, 3) + b * np.power(x, 2) + c * np.power(x, 1) + d + 500 * np.random.rand(len(x))


# Curve for polynomial 2x² -5x + 4 = 0
# mean = 0
# sd = 5
# s = np.random.normal(mean,sd, 100)
#
# x = s
# y =  2* x*x -5 * x + 4 +   100* np.random.rand(len(s))


class PolynomailRegression:
    def __init__(self, degree=2, alpha=0.0001, iterations=200, terms="", opt=""):
        self.degree = degree
        self.alpha = alpha
        self.iterations = iterations

        # reg terms
        self.terms = terms

        # optimization terms
        self.m = 0
        self.opt = opt
        ##for adam
        self.qm = 1
        self.qv = 1
        self.v = 0

    def normalize(self, X, a=0, b=1):
        return a + ((X - X.min()) * (b - a) / (X.max() - X.min()))

    def standrization(self, t):
        # for normally distributed  data
        X = t.copy()
        for i in range(X.shape[1] - 1):
            X[:, i + 1] = (X[:, i + 1] - np.mean(X[:, i + 1], axis=0)) / np.std(X[:, i + 1], axis=0)
        return X

    def transformation(self, X):
        transformation = np.ones(X.shape[0])
        for i in range(self.degree):
            temp = np.power(X, i + 1)
            transformation = np.vstack((transformation, temp))
        return transformation

    def regularization(self, w, B=0.5):
        t = 0
        if self.terms.lower() == "l1":
            for i in range(len(w)):
                t += math.fabs(w[i])
            return t * B
        elif self.terms.lower() == "l2":
            for i in range(len(w)):
                t += math.pow(w[i], 2)
            return math.sqrt(t) * B
        elif self.terms.lower() == "Lmax":
            return math.fabs(w.max())
        else:
            return t

    def optimizer(self, type, X, B=0.8, Bv=0.9):
        if type.lower() == "momentum":
            self.m = self.m * B + X
            print(self.m.max())
            return self.m
        elif type.lower() == "adam":
            self.qm = B * self.qm
            self.qv = Bv * self.qv
            self.m = self.m * B + (1 - B) * X
            self.v = Bv * self.v + (1 - Bv) * np.multiply(X, X)
            return self.m / (np.sqrt(self.v) + (10 ** -6)) * (math.sqrt(1 - self.qv) / 1 - self.qm)
        return X

    def train(self, X, y, term="", opt=""):
        self.opt = opt
        self.terms = term
        # tranforming data
        trans = self.transformation(X).T

        # normalizing Data in normal distribution
        norm = self.normalize(trans)
        # weights initialization
        self.w = np.ones(self.degree + 1)
        old = 0
        grad = []
        for i in range(self.iterations):

            # predicting values
            X_transform = self.transformation(X).T
            pred = np.dot(X_transform, self.w)

            # cost function
            cost = pred - y
            if self.iterations > 100:
                grad.append(cost.max())
            print(i, cost.max(), self.w)
            # updating weights
            self.w = self.w - self.alpha * (1 / X.shape[0]) * self.optimizer(self.opt, np.dot(norm.T,
                                                                                              cost)) + self.regularization(
                self.w, self.alpha * 10 ** 3)

        plt.plot(range(len(grad)), np.array(grad))
        plt.show()
        print(cost.max() * 10 ** -10)
        return self


model = PolynomailRegression(iterations=200000, degree=5, alpha=0.00000000000001)
weights = model.train(x, y, term="", opt="").w

y_ = 0
for i in range(weights.shape[0]):
    y_ = y_ + weights[i] * np.power(x, i)

print("real", f, e, d, c, b, a)
print("hypothesis", weights)

# print("difference", f - weights[0], e - weights[1], d - weights[2], c - weights[3], b - weights[4], a - weights[5])
# print("difference",  c - weights[1], b - weights[2], a - weights[3])

print("check result", (y_ - y).max())
plt.scatter(x, y_, color="red", label="generated")
plt.scatter(x, y, color="black", label="real")
plt.show()

# tips:
# 1. PASSING PARAMETERE DOESN'T COPY IT
# 2.use Normalization function to
# 3. difference between the normalization and standardization is important, but they can give the same model
# 4. overfitting can be sakved using large dataset "data points not features because features will increase complexity"
# or regularization technique

# problems:
# 1. coff of leading power is accurate, but the less the power the less the accuracy it gets
# 2. i need a very small alpha in high degree so some reason


# Generalized Linear Models:
#   this model can use transformation to do non-linear regression, which maps inputs to a different space and do a linear/classification in the hyper plane of that space,
#   the mapping is done to space that make the input a linear in the new space but in the original space it's a non-linear          example Φ(x) = x^3
#   this means if we have a quadratic function we map the quadratic input to a quadratic space that
#   make the input a linear and solve it. this will help us find almost all functions, and we can use any non-linear transformation like gaussian and etc
#   the change to the cost is that X will be equal to Φ(X) where Φ is vector that might have more than 1 non-linear function

#   personal thoughts:
#   i think that means that we use a Φ(x) to get to the y, so the equation will be y = W Φ(x) in which we just use a simple function and try to find the coefficient, and that's why choosing Φ() will be critical
#   because choosing the wrong function will not give us the solution, as an example if the true function is y = sin(x) , if we didn't use Φ(x) = sin(x) then we won't reach any solution:
#
#
#

