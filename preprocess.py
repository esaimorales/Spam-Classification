import numpy as np

# defines methods for data preprocessing

# standardize columns to all have mean 0 and unit variance
def standardize(features):
    result = (features - features.mean(axis=0))/features.std(axis=0)
    print result
    return result

# transform features using log(xij + 0.1)
def transform(data):
    return np.log(data + 0.1)

# binarize features using I(xij > 0)
def binarize(data):
    return data
