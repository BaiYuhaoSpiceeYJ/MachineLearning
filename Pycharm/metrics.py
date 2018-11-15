import numpy as np
from math import sqrt
def accuracy_score(y_true, y_predict):
    """计算y_true和y_predict之间的准确率"""
    assert len(y_true) == len(y_predict), \
        "the size of y_true must be equal to the size of y_predict"

    return np.sum(y_true == y_predict) / len(y_true)

def MSE(y_test, y_predict):
    """计算y_true和y_predict之间的MSE"""
    assert len(y_test) == len(y_predict), \
        "the size of y_true must be equal to the size of y_predict"

    return np.sum((y_test - y_predict)**2) / len(y_test)


def RMSE(y_test, y_predict):
    """计算y_true和y_predict之间的RMSE"""

    return sqrt(MSE(y_test, y_predict))


def MAE(y_test, y_predict):
    """计算y_true和y_predict之间的MAE"""

    return np.sum(np.absolute(y_test - y_predict)) / len(y_test)

def r2_score(y_test,y_predict):

    return 1-MSE(y_test,y_predict)/np.var(y_test)