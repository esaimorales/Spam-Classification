import numpy as np

from parse import get_features
from parse import get_classification

from preprocess import standardize
from preprocess import transform
from preprocess import binarize

# Perform Logistic Regression with Standardized features
print 'Logistic Regression with Standardized Features:'

print 'parsing...'

# parse train and test text files
train_x = get_features('spam_train.txt')
train_y = get_classification('spam_train.txt')

test_x = get_features('spam_test.txt')
test_y = get_classification('spam_test.txt')

print 'standardizing features... '

# standardize features
train_x = standardize(train_x)
test_x = standardize(test_x)