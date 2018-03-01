import numpy as np

# defines methods for data preprocessing

# standardize columns to all have mean 0 and unit variance
def standardize(features):
    result = (features - features.mean(axis=0))/features.std(axis=0)
    return result

# transform features using log(xij + 0.1)
def transform(data):
    return np.log(data + 0.1)

# binarize features using I(xij > 0)
def binarize(data):
    mu = data.mean(axis = 0)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i, j] >= mu[j]:
                data[i,j] = 1
            else:
                data[i,j] = 0

    return data
