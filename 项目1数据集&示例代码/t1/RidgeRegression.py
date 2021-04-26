import numpy as np
from LinearRegression import LinearRegression
from sklearn.model_selection import train_test_split


class RidgeRegression(LinearRegression):
    """
    岭回归模型
    """

    def __init__(self, L2=1):
        # [interception, x_1, x_2, x_3, ... x_n]
        super().__init__()
        self._L2 = L2

    def fit_normal(self, x_train, y_train):
        assert x_train.shape[0] == y_train.shape[0], "数据集有问题"
        x_train = self._data_arrange(x_train)
        self._theta = np.linalg.inv(x_train.T.dot(x_train) + np.eye(x_train.shape[1]).dot(self._L2)).dot(x_train.T).dot(
            y_train)
        return self

    def fit_cv(self, x_train, y_train, alphas, scoring='rmse'):
        """
        使用交叉验证自动搜寻最合适的L2
        """
        min_error = 10
        min_L2 = alphas[0]
        theta = self._theta
        for alpha in alphas:
            split_x_train, x_test, split_y_train, y_test = train_test_split(x_train, y_train, train_size=0.8)
            self._L2 = alpha
            self.fit_normal(split_x_train, split_y_train)
            error = self.score(x_test, y_test, scoring)
            if error < min_error:
                min_error = error
                min_L2 = alpha
                theta = self._theta
        self._L2 = min_L2
        self._theta = theta
        return self
