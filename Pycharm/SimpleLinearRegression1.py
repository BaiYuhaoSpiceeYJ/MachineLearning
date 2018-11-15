import numpy as np
#用for循环
class SimpleLinearRegression_1:

    def __init__(self):
        self.a_ = None
        self.b_ = None


    def fit(self,x_train,y_train):
        assert x_train.ndim ==1,\
            "can only solve single feature training data"
        assert len(x_train) == len(y_train),\
            "the size of x must be equal to y"

        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        num = 0.0
        d = 0.0
        for x,y in zip(x_train,y_train):
            num +=(x-x_mean)*(y-y_mean)
            d += (x-x_mean)**2

        self.a_ = num/d
        self.b_ = y_mean - self.a_ * x_mean
        return self

    def predict(self,x_predict):
        #x_predict为一个向量组
        assert x_predict.ndim == 1,\
            "it can only solve single feature training data"
        assert self.a_ is not None and self.b_ is not None,\
            "must fit before predict"

        return np.array([self._predict(x) for x in x_predict])

    def _predict(self,x_single):
        return self.a_*x_single+self.b_

    def __repr__(self):
        return "simpleLinearRegression1()"



