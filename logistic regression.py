# under construction


# mathematical derivation:

#   estimating the probability of each class:
#   Assumptions:
#   for number of classes we use multinomial distribution for P(C), and the data distribution for classes are gaussian
#   and each distribution have the same covariance "this mean the same data variance in all classes"

#   Posterior probability of class will give a sigmoid function for 2 classes or softmax for multiple classes, in general
#   softmax is a generalization of the sigmoid,
#   now using Argmax distribution is distribution that gives the highest probability of class using function f a value of 1 and 0 for everything
#   this degenerate distribution can be the approximation of the softmax, where changing the base of the softmax will make the difference between the
#   classes bigger, so the higher the base the bigger the difference, thus making the base limit to infinite the highest class will have a value of 1 and all
#   others will equal zero, "that's why it's called softmax because making the base smaller give us a probability for other classes, not just 1 and 0"
#
#   The parameters of the model:
#   the parameters are: 1. probability of each class 2. the mean of each class 3. the covariance matrix
#   for learning the 3 types of learning can be used, but the likelihood is te simplest and easiest to use, using the likelihood equation we can optimize and derive
#   each parameter to get the maximum, the mathematical expression will be ∏ [P(π N(x..) exp yn)(P(π N(x..) exp zn)... for each class], where whenever the data point
#   belonged to a classes the exp of the probability of the class will equal to 1 and all others will equal to zero.
#   as an example yn will equal to 1 and zn and all other classes will equal to 0 will equal to zero:
#   after taking the derivative and set it to zero for each parameter we get expressions to solve the parameters
#   parameter 1 will equal to the number of data point in class over the size of the data set
#   parameter 2 will equal to the expectation of the input of the class µ = ∑  y*x / Nc , where Nc is the number of data points in the class
#   parameter 3 ? :

#   "the proof is too long so i won't right it"
#   so the Posterior become the sigmoid function that predict the outcome of a class
#   and using the posterior only estimate the class not the parameters because the hypothesis is the class itself, so it's predicting the outcome
#
#   the separator of the classes is dependent on the covariance of the classes, it they are equal the separator will be a linear
#
#   in classification problems data is concentrated into points for each class, and because of the error we observe data in gaussian distribution
#
#
#
#
#
#   Q: does the Posterior and other equation for 1 class only? what about other classes?
#
#
#
#
#
#
