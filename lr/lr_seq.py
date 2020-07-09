"""Script for generation of artificial datasets."""
from typing import List

from lr import base


class LinearRegressionSequential(base.LinearRegression):
    """Replace."""

    def fit(self, X: List[float], y: List[float]) -> base.LinearRegression:
        """Replace."""
        n = len(X)
        x_sum = 0
        y_sum = 0
        x_squared_sum = 0
        x_y_mult_sum = 0
        for i in range(n):
            x_sum += X[i]
            y_sum += y[i]
            x_squared_sum += X[i] * X[i]
            x_y_mult_sum += X[i] * y[i]
        nominator = (n * x_y_mult_sum - x_sum * y_sum)
        a = nominator / (n * x_squared_sum - x_sum * x_sum)
        b = (y_sum / n) - a * (x_sum / n)
        self._coef = [b, a]
