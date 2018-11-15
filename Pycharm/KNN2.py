import numpy as np
from math import sqrt
from collections import Counter
from accuracy import accuracy_score

class KNNClassifier:
    def __init__(self,k):
        assert k>0,"k must be valid"
        self.k = k
        self._x_train = None#加_，为私有变量
        self._y_train = None

    def fit(self,x_train,y_train):
        assert x_train.shape[0] == y_train.shape[0], \
            "the size of x_train must be equal to y_train"
        assert self.k<=x_train.shape[0], \
            "the size of x_train must be at least k"
        self._x_train=x_train
        self._y_train=y_train
        return self

    def predict(self,x_predict):#x_predict为待预测数据
        assert self._x_train is not None and self._y_train is not None,\
            "must fit before predict"
        assert x_predict.shape[1]==self._x_train.shape[1],\
            "the feature number of predict must be equal to x_train"
        y_predict = [self._predict(x) for x in x_predict]
        return np.array(y_predict)

    def _predict(self,x):
        distances = [sqrt(np.sum((xtrain - x) ** 2)) for xtrain in self._x_train]
        k_nearest = np.argsort(distances)
        k_top_y = [self._y_train[i] for i in k_nearest[:self.k]]
        vote = Counter(k_top_y)
        return vote.most_common(1)[0][0]

    def __repr__(self):
        return "kNN(k=%d)" %self.k

    def score(self,x_test,y_test):
        result = self.predict(x_test)
        return accuracy_score(y_test,result)
