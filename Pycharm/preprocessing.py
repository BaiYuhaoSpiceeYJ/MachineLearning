import numpy as np

class StandardScaler:

    def __init__(self):
        self.mean_= None
        self.scale_= None

    def fit(self,x):#获得均值和方差
        assert x.ndim==2,"the dimension must be 2"#只能传二维数据

        self.mean_ = np.array([np.mean(x[:,i]) for i in range(x.shape[1])])
        self.scale_ = np.array([np.std(x[:,i]) for i in range(x.shape[1])])
        return self

    def transform(self,x):#将x根据数据进行归一化
        assert x.ndim == 2, "the dimension must be 2"  # 只能传二维数据
        assert self.mean_ is not None and self.scale_ is not None,\
            "must fit before transform"
        assert x.shape[1]==len(self.mean_),\
            "the feaature of x must equal to mean_ and std_"
        resX = np.empty(x.shape,dtype=float)
        for col in range(x.shape[1]):
            resX[:col]=(x[:,col] - self.mean_[col])/self.scale_[col]
        return resX