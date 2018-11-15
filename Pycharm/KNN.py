import numpy as np
from math import sqrt
from collections import Counter

def KNN_classify(k,x_train,y_train,s):
    assert  1<=k<=x_train.shape[0],"k must be valid"#try catch throw
    assert x_train.shape[0]==y_train.shape[0],\
        "the size of x_train must be equal to y_train"
    assert x_train.shape[1]==s.shape[0],\
        "the feature number of x must be equal to x_train"

    distances = [sqrt(np.sum((x-s)**2)) for x in x_train]
    k_nearest = np.argsort(distances)
    k_top_y = [y_train[i] for i in k_nearest[:k]]
    vote = Counter(k_top_y)

    return vote.most_common(1)[0][0]

