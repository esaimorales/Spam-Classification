import numpy as np

from parse import get_features
from parse import get_classification
from parse import add_ones

from preprocess import standardize
from preprocess import transform
from preprocess import binarize

from functions import logistic_regression
from functions import predict_y
from functions import sigmoid
from functions import stabilize
from functions import get_error

# Perform Logistic Regression with different preprocessing techniques

# --------------------------------------------------- #
# -------------- Standardized Features -------------- #
# --------------------------------------------------- #

print 'Logistic Regression with Standardized Features:'

# print 'parsing...'

# parse train and test text files
train_x = get_features('spam_train.txt')
train_y = get_classification('spam_train.txt')

test_x = get_features('spam_test.txt')
test_y = get_classification('spam_test.txt')

# print 'standardizing features...'

# standardize features
train_x = standardize(train_x)
test_x = standardize(test_x)

# add 1 y-intercept column
train_x = add_ones(train_x)
test_x = add_ones(test_x)

# print 'calculating weights...'

# find W for logistic regression with gradient descent
w = logistic_regression(train_x, train_y)

# print 'predicting...'

# make predictions
train_predictions = predict_y(train_x, w)
test_predictions = predict_y(test_x, w)

# apply sigmoid
train_predictions = sigmoid(train_predictions)
test_predictions = sigmoid(test_predictions)

# stabilize predictions
train_predictions = stabilize(train_predictions)
test_predictions = stabilize(test_predictions)

# calculate error
error_rate = get_error(train_predictions, train_y)
print 'Train - Error Rate:', error_rate

error_rate = get_error(test_predictions, test_y)
print 'Test  - Error Rate:', error_rate

# -------------------------------------------------- #
# ------------- Transformed Features --------------- #
# -------------------------------------------------- #

print 'Logistic Regression with Transformed Features:'

# print 'refreshing dataset...'

# parse train and test text files
train_x = get_features('spam_train.txt')
train_y = get_classification('spam_train.txt')

test_x = get_features('spam_test.txt')
test_y = get_classification('spam_test.txt')

# print 'transforming features...'

# standardize features
train_x = transform(train_x)
test_x = transform(test_x)

# add 1 y-intercept column
train_x = add_ones(train_x)
test_x = add_ones(test_x)

# print 'calculating weights...'

# find W for logistic regression with gradient descent
w = logistic_regression(train_x, train_y)

# print 'predicting...'

# make predictions
train_predictions = predict_y(train_x, w)
test_predictions = predict_y(test_x, w)

# apply sigmoid
train_predictions = sigmoid(train_predictions)
test_predictions = sigmoid(test_predictions)

# stabilize predictions
train_predictions = stabilize(train_predictions)
test_predictions = stabilize(test_predictions)

# calculate error
error_rate = get_error(train_predictions, train_y)
print 'Train - Error Rate:', error_rate

error_rate = get_error(test_predictions, test_y)
print 'Test  - Error Rate:', error_rate

# ------------------------------------------------ #
# -------------- Binarized Features -------------- #
# ------------------------------------------------ #

print 'Logistic Regression with Binarized Features:'

# print 'refreshing dataset...'

# parse train and test text files
train_x = get_features('spam_train.txt')
train_y = get_classification('spam_train.txt')

test_x = get_features('spam_test.txt')
test_y = get_classification('spam_test.txt')

# print 'binarizing features...'

# standardize features
train_x = binarize(train_x)
test_x = binarize(test_x)

# add 1 y-intercept column
train_x = add_ones(train_x)
test_x = add_ones(test_x)

# print 'calculating weights...'

# find W for logistic regression with gradient descent
w = logistic_regression(train_x, train_y)

# print 'predicting...'

# make predictions
train_predictions = predict_y(train_x, w)
test_predictions = predict_y(test_x, w)

# apply sigmoid
train_predictions = sigmoid(train_predictions)
test_predictions = sigmoid(test_predictions)

# stabilize predictions
train_predictions = stabilize(train_predictions)
test_predictions = stabilize(test_predictions)

# calculate error
error_rate = get_error(train_predictions, train_y)
print 'Train - Error Rate:', error_rate

error_rate = get_error(test_predictions, test_y)
print 'Test  - Error Rate:', error_rate
