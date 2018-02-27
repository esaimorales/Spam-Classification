import numpy as np
import random

# Logistic regression and helper functions

# defines sigmoid function
def sigmoid(value):
    value = 1 / (1+ np.exp(-value))
    return value

# returns mean error rate
def mean_error_rate(predicted, target):
    return ((predicted - target) ** 2).mean()

def logistic_regression(X, Y):

    # set alpha (step-size)
    alpha = 0.00001

    # start W_0 at random value
    W_previous = np.full((X.shape[1], 1), random.uniform(0,1))



    return Y
