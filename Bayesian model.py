# under construction


#   This model is based on bayesian rule

#   P(H|e) = P(e|H) * P(H) / P(e)
#   P(H|e) => Posterior probability:
#       it is the updated prior distribution after taking evidence "information" in consideration
#   P(e|H) => Likelihood probability:
#       given a hypothesis, it is the probability of the sample in normal distribution of the hypothesis for the same input
#   P(H) => Prior Probability:
#       it is used to reduce uncertainty, it's initial belief of the hypothesis, example we believe that the probability of flipping a coin
#       and getting head is 50%, it's an initial belief, the domain expert should select.
#   P(e) => Data Probability "Normalization constant":
#       because we calculate the probability for the same dataset the probability of the dataset doesn't change, so the P(e) can be considered as a constant for normalize
#       and make the range of the probability between [0,1]

#   Predicting: after getting the Posterior probability of many hypothesis to predict the outcome of P(X|e) where X is the prediction
#       can be done by sum the product of the outcome probability with the Posterior probability of each model      P(X|e)  =  Σ  P(X|hi) * P(hi|e)
#
#   Bayesian learning:
#   pros:   1. is optimal "as long as the prior is correct" and no other prediction is correct more than often than Bayesian one
#           2. it's immune to overfitting because it doesn't consider one hypothesis but all hypothesis space, this happens because finding
#           the optimal hypothesis might overfit the data, but in Bayesian it sees how every hypothesis fits the data and compute posterior
#           so even if one hypothesis overfit the other will counterbalance it
#   cons    1. hypothesis space can be very big or infinitely large "solved using 2 optimization technique"
#           2. sum over hypothesis often intractable??????
#   Solving the Bayesian learning problem:
#       1.  MAP Maximum posteriori: taking the highest Posterior probability of hypothesis, but it's less accurate, but it's CONVERGE as data increase,but it might overfit
#           for control overfitting we use prior to penalize complex hypothesis, so simple hypothesis have high prior probability and complex have low
#           P(X|e) = P(X|h) with the highest P(h|e)
#       cons: 1.finding the best hypothesis is optimization problem which is NP 2. overfitting issue
#       2.  ML Maximum Likelihood: taking the highest likelihood probability, so just find the hypothesis that fits the data best
#       cons: 1. overfitting and less accurate 2. it's an optimization problem and intractability problem

#   having a very large data "limit to inf" will keep reducing uncertainty and will be more confident
#   in some hypothesis and on limit will converge on one or few hypothesis that is equally
#   good, so if the mass of the model concentrated on one hypothesis the MAP and ML will give the same answer, so with infinite amount of data the 3 are equivalent
#   so how much data do I need? it depends on the hypothesis space "learning theory is dedicated for this Q"


#   Q: How can we have probability over functions
#   Q: isn't MAP and ML an optimization problem ?
