import numpy as np
import random
import sys

# Logistic regression and helper functions

# defines sigmoid function
def sigmoid(value):
    value = 1 / (1+ np.exp(-value))
    return value

# returns mean error rate
def mean_error_rate(predicted, target):
    return ((predicted - target) ** 2).mean()

def get_error(predicted, target):
    a = predicted - target
    miss_count = np.count_nonzero(a)
    # print 'Miss:', miss_count
    # print 'Total:', target.shape[0]
    return float(miss_count/float(target.shape[0]))

# defines logistic regression with gradient descent
# follows lgistic regression function:
# W_t+1 = W_t + a X.T (Y - S(W.T Xi))
def logistic_regression(X, Y):
    # set alpha (step-size)
    alpha = 0.00001

    # start W_t at random value
    W_previous = np.full((X.shape[1], 1), random.uniform(0,1))

    # find S value and set W_t+1
    S = sigmoid(np.dot(X, W_previous))
    W_current = W_previous + np.dot((alpha * X.T), (Y-S))

    # set diff and threshold
    diff, epsilon = 1, 0.0001

    while abs(diff) > epsilon:
        # reset W_t
        W_previous = W_current

        # recalculate W_t+1
        S = sigmoid(np.dot(X, W_previous))
        W_current = W_previous + np.dot((alpha * X.T), (Y-S))
        diff = np.amin(W_current - W_previous)

    return W_current

# make prediction
def predict_y(X, w):
    w = w.T
    y = np.dot(w, (X.T))
    return y.T

# puts results in [0,1] domain 
def stabilize(p):
    return np.where(p > 0.5, 1, 0)
