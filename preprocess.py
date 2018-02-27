import numpy as np

# defines methods for data preprocessing

# standardize columns to all have mean 0 and unit variance
def standardize(features):
    # mu = (np.mean(np.matrix(features), axis = 0)).T
    # print mu
    # print features.mean(axis=0)
    result = (features - features.mean(axis=0))/features.std(axis=0)
    return result

# transform features using log(xij + 0.1)
def transform(data):
    return np.log(data + 0.1)

# binarize features using I(xij > 0)
def binarize(data):
    return data 
