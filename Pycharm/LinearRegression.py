import numpy as np
from metrics import r2_score
#多元线性回归
class LinearRegression:

    def __init__(self):
        self.coef_ = None#特征对应的系数
        self.intercept_ = None#截距
        self._theta = None

    def fit_normal(self,x_train,y_train):
        assert x_train.shape[0] == y_train.shape[0],\
            "the size of x must be equal to y"
        x_b = np.hstack([np.ones((len(x_train),1)),x_train])#对x加上一列
        self._theta = np.linalg.inv(x_b.T.dot(x_b)).dot(x_b.T).dot(y_train)
        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]
        return self

    def fit_gd(self,x_train,y_train,eta=0.01,n_iters=1e-4):
        assert x_train.shape[0] == y_train.shape[0], \
            "the size of x must be equal to y"

        def J(theta, x_b, y_train):  # 损失函数
            try:
                return np.sum((y_train - x_b.dot(theta)) ** 2) / len(x_b)
            except:
                return float('inf')

        def dJ(theta, x_b, y_train):  # 导数矩阵
            #for 循环方法
            #res = np.empty(len(theta))
            #res[0] = np.sum(x_b.dot(theta) - y_train)
            #for i in range(1, len(theta)):
                #res[i] = (x_b.dot(theta) - y_train).dot(x_b[:, i])
            #return res * 2 / len(x_b)
            return x_b.T.dot(x_b.dot(theta)-y_train)*2./len(x_b)#向量化方法

        def dJ_debug(theta, x_b, y, epsilon=0.01):
            res = np.empty(len(theta))
            for i in range(len(theta)):
                theta_1 = theta.copy()  # 对某一个特征的偏导数
                theta_1[i] += epsilon
                theta_2 = theta.copy()
                theta_2[i] -= epsilon
                res[i] = (J(theta_1, x_b, y) - J(theta_2, x_b, y)) / (2 * epsilon)
            return res  # 模拟的导数值

        def gradient_descent(x_b, y_train, initial_theta, eta, n_iters=10000, epsilon=1e-8,):
            theta = initial_theta
            i_iter = 0
            while i_iter < n_iters:
                gradiant = dJ(theta, x_b, y_train)
                last_theta = theta
                theta = theta - eta * gradiant
                if (abs(J(theta, x_b, y_train) - J(last_theta, x_b, y_train)) < epsilon):
                    break
                i_iter += 1

            return theta

        x_b = np.hstack([np.ones((len(x_train), 1)), x_train])
        initial_theta = np.zeros(x_b.shape[1])
        self._theta = gradient_descent(x_b, y_train, initial_theta, eta)
        self.intercept_ = self._theta[0]
        self.coef_=self._theta[1:]
        return self

    def fit_sgd(self, X_train, y_train, n_iters=50, t0=5, t1=50):#简单线性回归
        """根据训练数据集X_train, y_train, 使用梯度下降法训练Linear Regression模型"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"
        assert n_iters >= 1

        def dJ_sgd(theta, X_b_i, y_i):
            return X_b_i * (X_b_i.dot(theta) - y_i) * 2.

        def sgd(X_b, y, initial_theta, n_iters=5, t0=5, t1=50):

            def learning_rate(t):
                return t0 / (t + t1)

            theta = initial_theta
            m = len(X_b)
            for i_iter in range(n_iters):
                indexes = np.random.permutation(m)
                X_b_new = X_b[indexes, :]
                y_new = y[indexes]
                for i in range(m):
                    gradient = dJ_sgd(theta, X_b_new[i], y_new[i])
                    theta = theta - learning_rate(i_iter * m + i) * gradient

            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.random.randn(X_b.shape[1])
        self._theta = sgd(X_b, y_train, initial_theta, n_iters, t0, t1)

        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    def fit_mbgd(self, X_train, y_train, n_iters=10000, n_random=30,t0=5, t1=50):#批量梯度下降法
        """根据训练数据集X_train, y_train, 使用梯度下降法训练Linear Regression模型"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"
        assert n_iters >= 1
        assert n_random <= X_train.shape[0],\
            "n_random can not bigger than the number of X_train"

        def dJ_mbgd(theta, x_b_mb, y_mb, n_random):  # 导数矩阵
            return x_b_mb.T.dot(x_b_mb.dot(theta) - y_mb) * 2. / n_random

        def mbgd(x_b, y, initial_theta, n_iters,n_random, t0, t1):

            def learning_rate(t):
                return t0 / (t + t1)
                #return 0.01

            theta = initial_theta
            for cur_iter in range(n_iters):
                shuffled_indexes = np.random.permutation(len(x_b))
                mbgd_index = shuffled_indexes[:n_random ]
                x_b_mb = x_b[mbgd_index]
                y_mb = y[mbgd_index]
                gradient = dJ_mbgd(theta, x_b_mb, y_mb, n_random)
                theta = theta - learning_rate(cur_iter) * gradient
            return theta

        x_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.zeros(x_b.shape[1])
        self._theta = mbgd(x_b, y_train, initial_theta, n_iters,n_random, t0, t1)

        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    def predict(self,x_predict):
        assert self.intercept_ is not None and self.coef_ is not None,\
            "must fit before predict"
        assert x_predict.shape[1] == len(self.coef_),\
            "the feature of x_predict must be equal to x_train"
        x_b = np.hstack([np.ones((len(x_predict),1)),x_predict])
        return x_b.dot(self._theta)

    def score(self,x_test,y_test):
        y_predict = self.predict(x_test)
        return r2_score(y_test,y_predict)

    def __repr__(self):
        return "LinearRegression"