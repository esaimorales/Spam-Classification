import numpy as np

# defines methods for parsing the train and test datasets

# returns numpy array of feature values
def get_features(file_name):
    with open(file_name) as f:
        return np.array([[float(val) for val in line.split(',')[:57]] for line in f])

# returns numpy array of classication
def get_classification(file_name):
    with open(file_name) as f:
        return np.array([[float(val) for val in line.split(',')[57:]] for line in f])
