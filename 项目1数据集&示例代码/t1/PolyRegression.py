from LinearRegression import LinearRegression
import sklearn.preprocessing as sp
from sklearn.linear_model import LinearRegression as lr
from sklearn.model_selection import train_test_split


class PolyRegression(LinearRegression):
    """
    多项式回归模型
    """

    def __init__(self):
        super(PolyRegression, self).__init__()
        self.degree = None
        self._poly = None
        self._lr = lr()

    def fit_normal(self, x_train, y_train, degree=10):
        self.degree = degree
        self._poly = sp.PolynomialFeatures(degree)
        self._lr.fit(self._poly.fit_transform(x_train), y_train)
        return self

    def fit_cv(self, x_train, y_train, degrees, scoring='rmse'):
        min_error = 10
        min_degree = degrees[0]
        for degree in degrees[1:]:
            split_x_train, x_test, split_y_train, y_test = train_test_split(x_train, y_train, train_size=0.8)
            self.fit_normal(split_x_train, split_y_train, degree)
            error = self.score(x_test, y_test, scoring)
            if error < min_error:
                min_error = error
                min_degree = degree
        self.fit_normal(x_train, y_train, min_degree)

    def predict(self, x_predict):
        return self._lr.predict(self._poly.fit_transform(x_predict))

    def score(self, x, y, method='rmse'):
        return super()._score(self.predict(x), y, method)
