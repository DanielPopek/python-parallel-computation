"""Script for generation of artificial datasets."""
from typing import List
from multiprocessing import Pool
from lr import base


class LinearRegressionProcess(base.LinearRegression):
    """Replace."""

    def mult_summing_function(self, first_collection, second_collection):
        """Replace."""
        sum = 0
        n = len(first_collection)
        if second_collection is not None:
            for i in range(n):
                sum += first_collection[i] * second_collection[i]
        else:
            for i in range(n):
                sum += first_collection[i]
        return sum

    def fit(self, X: List[float], y: List[float]) -> base.LinearRegression:
        """Replace."""
        n = len(X)
        p = Pool(4)
        x_sum = p.apply_async(self.mult_summing_function, (X, None))
        y_sum = p.apply_async(self.mult_summing_function, (y, None))
        x_squared_sum = p.apply_async(self.mult_summing_function, (X, X))
        x_y_mult_sum = p.apply_async(self.mult_summing_function, (X, y))

        p.close()

        x_sum = x_sum.get()
        y_sum = y_sum.get()
        x_squared_sum = x_squared_sum.get()
        x_y_mult_sum = x_y_mult_sum.get()

        nominator = (n * x_y_mult_sum - x_sum * y_sum)
        a = nominator / (n * x_squared_sum - x_sum * x_sum)
        b = (y_sum / n) - a * (x_sum / n)
        self._coef = [b, a]
