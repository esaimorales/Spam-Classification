import numpy as np

# Defines methods for parsing the train and test datasets

# returns numpy array of feature values
def get_features(file_name):
    with open(file_name) as f:
        return np.array([[float(val) for val in line.split(',')[:57]] for line in f])

# returns numpy array of classication
def get_classification(file_name):
    with open(file_name) as f:
        return np.array([[float(val) for val in line.split(',')[57:]] for line in f])

# adds a 1 to the end of every row (instance)
def add_ones(array):
    ones = np.ones((array.shape[0], 1))
    return np.hstack((array, ones))
