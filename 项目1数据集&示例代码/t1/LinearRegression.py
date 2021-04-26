import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

'''
线性回归模型
'''


class LinearRegression:
    """
    线性回归模型
    """

    def __init__(self):
        # [interception, x_1, x_2, x_3, ... x_n]
        self._theta = None  # 回归系数矩阵

    def fit_normal(self, x_train, y_train):
        assert x_train.shape[0] == y_train.shape[0], "数据集有问题"
        x_b = self._data_arrange(x_train)
        self._theta = np.linalg.inv(x_b.T.dot(x_b)).dot(x_b.T).dot(y_train)
        return self

    def predict(self, x_predict):
        x_predict = self._data_arrange(x_predict)
        return x_predict.dot(self._theta)

    def score(self, x, y, method='rmse'):
        return self._score(self.predict(x), y, method)

    def _score(self, y_pred, y, method='rmse'):
        error = -1
        if method == 'rmse':
            error = np.sqrt(mean_squared_error(y, y_pred))
        if method == 'r2':
            error = r2_score(y, y_pred)
        return error

    '''
    为原有数据添加截距表示
    '''

    @staticmethod
    def _data_arrange(data):
        return np.hstack([np.ones((len(data), 1)), data])
