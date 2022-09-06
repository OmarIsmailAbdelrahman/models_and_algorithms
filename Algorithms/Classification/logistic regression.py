# under construction
import sys
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

np.set_printoptions(threshold=sys.maxsize)


class logistic:
    def __init__(self, alpha=0.01, multi_class="", max_iter=1000, penalty=""):
        self.W = None
        self.alpha = alpha
        self.multi_class = multi_class
        self.max_iter = max_iter
        self.penalty = penalty

    def sigmoid(self, z, base=math.e):
        tmp = 1 / (1 + np.exp(-z.dot(self.W)))
        return tmp

    def predict(self, z, base=math.e):
        tmp = 1 / (1 + np.exp(-z.dot(self.W)))
        for i in range(tmp.shape[0]):
            if tmp[i] > 0.5:
                tmp[i] = 1
            else:
                tmp[i] = 0
        return tmp

    def standrization(self, X):
        temp = np.ones(shape=X.shape)
        for i in range(1, X.shape[1]):
            # print(i,X[:,i])
            temp[:, i] = (X[:, i] - np.mean(X[:, i])) / np.std(X[:, i])
        return temp

    def regularization(self, lamb):
        if self.penalty.lower() == "l2":
            return lamb * np.sum(self.W ** 2)
        elif self.penalty.lower() == "l1":
            tmp = np.fabs(self.W)
            return tmp.sum()
        return 0

    def gradientDecent(self, batch, batchy):
        h = self.sigmoid(batch)
        cost = h - batchy
        return self.W - batch.T.dot(cost) * self.alpha

    def train(self, X, y, lamb=0.2):

        # initializing the weights
        self.W = np.ones(X.shape[1] + 1)
        # adding ones column to data
        X = np.vstack((np.ones(X.shape[0]), X.T)).T

        # normalization Data
        # self.standrization(X)
        for i in range(self.max_iter):

            # this is batch, i think :D
            s1 = min(((i * int(X.shape[0] / 10)) % X.shape[0]), (((i + 1) * int(X.shape[0] / 10)) % X.shape[0]))
            s2 = max(((i * int(X.shape[0] / 10)) % X.shape[0]), (((i + 1) * int(X.shape[0] / 10)) % X.shape[0]))
            if s1 < X.shape[0] / 10:
                s1 = s2 - int(X.shape[0] / 10)
            batch = X[s1:s2, :]
            batchy = y[s1:s2]

            # print(batch.shape, batchy.shape, s1, s2)
            # print(i,y_)
            gradient = self.gradientDecent(batch, batchy)
            print(gradient.sum())
            self.W = gradient + self.regularization(0.001)


df = pd.read_csv("diabetes.csv")
# df["Gender"] = df.apply(LabelEncoder().fit_transform)["Gender"]
X = df.drop(["Outcome"], axis=1).to_numpy()
y = df["Outcome"].to_numpy()
model = logistic(alpha=0.1, penalty="L2", max_iter=10000)
X = model.standrization(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.train(X_train, y_train)
X_test = np.vstack((np.ones(X_test.shape[0]), X_test.T)).T
print(model.W)
false = 0
t = model.predict(X_test)
for i in range(t.shape[0]):
    if t[i] != y_test[i]:
        false += 1
print("accuracy", false, "of", t.shape[0], 1 - false / y_test.shape[0])

#   mathematical  of MIXTURE OF GAUSSIAN DISTRIBUTION  :
#   this model considers that the distribution of each class have a gaussian distribution.
#
#   estimating the probability of class:
#       using the posterior distribution P(C|X) gives the probability of a class for given data, the prior distribution will equal multinomial, in which for a specific class will equal π, and the likelihood
#       will equal to P(X|C) which will be a normal distribution, and the normalization constant will equal to the submission of likelihood of all classes  :
#               P(Ci|X) = π * Likelihood(Ci) / ∑ Likelihood(C)
#       changing the equation will give us the sigmoid function for 2 classes or softmax for multiple classes, in general softmax is a general0ization of the sigmoid.
#       the exponent in the function is linear in X, so the Posterior probability can be writen as a sigmoid function that have WX+w0 form,
#       P(C|X) = σ(WX+w0)       so that's why we use sigmoid function because it came from the calculating the posterior distribution
#   Assumptions:
#   for number of classes we use multinomial distribution for P(C), and the data distribution for classes are gaussian
#   and each distribution have the same covariance "this mean the same data variance in all classes"
#   ""this predict the outcome not the parameters because the hypothesis is the class so we are trying to calculate the probability of class for a given data""
#
#   ArgMax:
#       is function gives the highest probability of class using function f a value of 1 and 0 for everything
#       this degenerate distribution can be the approximation of the softmax, where changing the base of the softmax will make the difference between the
#       classes bigger, so the higher the base the bigger the difference, thus making the base limit to infinite the highest class will have a value of 1 and all
#       others will equal zero, "that's why it's called softmax because making the base smaller give us a probability for other classes, not just 1 and 0"
#
#
#   Estimating parameters of the model:
#       using the likelihood maximization we get a function that is convex, this mean that there is a global maxima for the probability
#       the parameters are: 1. probability of each class P(C) 2. the mean of each class  3. the covariance matrix
#       for learning the 3 types of learning can be used, but the likelihood is te simplest and easiest to use, using the likelihood equation we can optimize and derive
#       each parameter to get the maximum, the mathematical expression will be ∏ [P(π N(x..) exp yn)(P(π N(x..) exp zn)... for each class], where whenever the data point
#       belonged to a classes the exp of the probability of the class will equal to 1 and all others will equal to zero.
#       as an example yn will equal to 1 and zn and all other classes will equal to 0 will equal to zero:
#
#   after taking the derivative and set it to zero for each parameter we get expressions to solve the parameters.
#   parameter 1 will equal to the number of data point in class over the size of the data set
#   parameter 2 will equal to the expectation of the input of the class µ = ∑  y*x / Nc , where Nc is the number of data points in the class
#   parameter 3 ? :
#   the separator of the classes is dependent on the covariance of the classes, if they are equal the separator will be a linear


#   Logistic Regression:
#   logistic regression is a generalization of GMM, because it doesn't need to be gaussian distribution, they can follow Exponential Family ["Gaussian","exponential","Gamma","Beta",...etc]
#   The form of this family is P(x,theta) = exp(theta.T * T(x) - A(theta) + B(x)) we can see that it has the exponential characteristics, the difference happens in the A and B function.
#
#   maximization function:
#   for the Exponential family the POSTERIOR will also equal to sigmoid function so we done need to calculate it and go straight to calculating the posterior which help us to change the Weights of the sigmoid function
#   Using likelihood to find the best Wights for the posterior, using the posterior to predict the outcome and using the yi that denote the class, the equation for single data point and two classes
#       w* = argmax σ(WX)^y ( 1 - σ(WX)) ^ 1 - y
#       the y will choose which probability will be considered for each data point, so each data point will return the probability that for a
#       given weights the probability that this point will be in the class is equal to it, and to maximize we consider the entire dataset
#       w* = argmax ∏  σ(WX)^yn ( 1 - σ(WX)) ^ (1 - yn) , using log then derivative of weights, it will equal to σ(a)(1-σ(a))
#       from this expression the gradient function will equal ∑ [σ(Wx)-y]x
#       and because we can't isolate the W here we can't find a closed form solution for it so we use gradient or newton's method instead

#   Newton method:
#       is method that is much faster and have fewer steps and doesn't need step size
#       knowing that the Posterior distribution is the same  sigmoid function in Exponential Family, we can right away use likelihood maximization to get the right weights for the sigmoid function
#           formula: W <= W - inverse(H) * gradient              H = the second derivative matrix that pairs every derivative of weight together and the diagonal is the square of gradients
#           H = X R X           R = diag(  σ1(1-σ1) , σ2(1-σ2) ..... σn(1-σn)) where σi = σ(Wxi)
#       cons: the technique might over fit [logistic regression in general] because it will reach the global optimal,
#       and this will make the classification weights reach infinite values because it wants to reach 1, solving this by regularization terms H = X R X  + λI
#


#
#
#   Q: does the Posterior and other equation for 1 class only? what about other classes?
#   Q: regularization in newtons is posterior distribution?
#
#
#
#
#   notes: logistic regression solve a classification problem using regression because we compute the Posterior that is equal to 0 to 1, so it's a continuous values
